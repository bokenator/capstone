import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        zscore: Z-score of spread array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic sanity checks
    if i < 0:
        return (np.nan, 0, 0)

    # Read z-score and hedge ratio at this bar
    if i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # If we don't have valid indicators, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Prices
    price_a = close_a[i]
    price_b = close_b[i]
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Number of shares for each leg based on fixed notional per leg
    shares_a = float(notional_per_leg) / float(price_a)
    shares_b = (float(notional_per_leg) / float(price_b)) * float(hr)

    # Determine desired positions for both assets
    desired_a = None
    desired_b = None

    # Stop-loss: if z-score beyond stop_threshold, close positions
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entry signals
        if z > entry_threshold:
            # Short A, Long B (hedge_ratio units)
            desired_a = -shares_a
            desired_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            desired_a = +shares_a
            desired_b = -shares_b
        else:
            # Exit on crossing the exit_threshold (typically 0.0)
            crossed = False
            if i > 0 and not np.isnan(zscore[i - 1]):
                z_prev = zscore[i - 1]
                # Check for crossing of the threshold
                if (z_prev > exit_threshold and z <= exit_threshold) or (
                    z_prev < exit_threshold and z >= exit_threshold
                ):
                    crossed = True

            if crossed:
                desired_a = 0.0
                desired_b = 0.0
            else:
                # No action for this bar
                return (np.nan, 0, 0)

    # Map desired position to current column
    if col == 0:
        desired = float(desired_a)
    else:
        desired = float(desired_b)

    # Compute delta (order size in shares)
    delta = desired - float(pos)

    # If change is negligible, do nothing
    if np.isclose(delta, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return order as amount (shares)
    # direction=0 allows both long and short
    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible and accepts either pandas DataFrames with a
    'close' column (as specified in DATA_SCHEMA) or raw numpy arrays / pandas
    Series containing close prices. It returns numpy arrays aligned to the
    input length.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or a 1D array/Series
                 of close prices.
        asset_b: DataFrame with 'close' column for Asset B or a 1D array/Series
                 of close prices.
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close arrays from DataFrame or accept arrays/Series directly
    def _extract_close(x):
        if isinstance(x, (np.ndarray, pd.Series)):
            arr = np.asarray(x)
        elif isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame must contain 'close' column as per DATA_SCHEMA")
            arr = x['close'].values
        else:
            raise TypeError("asset_a/asset_b must be a DataFrame, Series, or ndarray")
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Hedge ratio (rolling OLS slope): initialize with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS using stats.linregress for each window
    # Place the slope at the last index of the window (i)
    start_idx = int(hedge_lookback)
    for i in range(start_idx, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove NaNs in the window
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            # Not enough data points to regress
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(hedge_ratio)
    spread[valid] = close_a[valid] - hedge_ratio[valid] * close_b[valid]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Avoid division by zero
    zscore = (spread_series - spread_mean) / spread_std
    zscore_vals = zscore.values.astype(float)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore_vals,
    }
