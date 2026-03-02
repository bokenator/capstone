import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


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

    # Basic validation
    if i < 0:
        return (np.nan, 0, 0)

    # Get indicators at current bar
    # Safe indexing for arrays (they should be 1D numpy arrays)
    try:
        z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    # If zscore is NaN, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices
    try:
        price_a = float(close_a[i])
        price_b = float(close_b[i])
    except Exception:
        return (np.nan, 0, 0)

    # Validate prices
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Hedge ratio
    try:
        h = float(hedge_ratio[i])
    except Exception:
        h = np.nan

    # If hedge ratio is not available, do nothing
    if np.isnan(h):
        return (np.nan, 0, 0)

    # Determine target sizes based on fixed notional per leg
    # Number of shares for asset A (base leg)
    shares_a = notional_per_leg / price_a
    # Number of shares for asset B scaled by hedge ratio
    shares_b = (notional_per_leg / price_b) * h

    # Stop-loss: if |zscore| > stop_threshold -> close any existing position
    if np.isfinite(z) and abs(z) > stop_threshold:
        # If there's an open position, close it by returning opposite of current position
        if pos != 0.0:
            # Return order in amount (shares) to close current position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Exit condition: z-score crosses exit_threshold (usually 0.0)
    prev_z = None
    if i > 0:
        prev_z = zscore[i - 1]

    crossed_zero = False
    if prev_z is not None and not np.isnan(prev_z):
        # Crossing logic: previous positive and now <= exit_threshold OR previous negative and now >= exit_threshold
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed_zero = True

    if crossed_zero:
        if pos != 0.0:
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry logic
    # When z > entry_threshold: Short A, Long B
    # When z < -entry_threshold: Long A, Short B
    target = None
    if z > entry_threshold:
        # Short A (negative), Long B (positive)
        if col == 0:
            target = -shares_a
        else:
            target = shares_b
    elif z < -entry_threshold:
        # Long A (positive), Short B (negative)
        if col == 0:
            target = shares_a
        else:
            target = -shares_b
    else:
        # No entry signal
        return (np.nan, 0, 0)

    # Compute required size to reach target (in shares)
    size = float(target - pos)

    # If size is effectively zero, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Place order by shares (Amount)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either pandas DataFrames with a 'close' column or 1D numpy arrays / pandas Series
    representing close prices.

    Args:
        asset_a: DataFrame or 1D array/Series with 'close' prices for Asset A
        asset_b: DataFrame or 1D array/Series with 'close' prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close price arrays from inputs (support both DataFrame and ndarray/Series)
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame inputs must contain a 'close' column")
            arr = x['close'].values
        elif isinstance(x, (pd.Series, np.ndarray, list)):
            arr = np.asarray(x)
        else:
            raise TypeError("asset inputs must be DataFrame, Series, ndarray or list")
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError('asset_a and asset_b must have the same length')

    n = close_a.shape[0]

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression to compute hedge ratio (slope of regression of A on B)
    # For each window ending at index i (exclusive), compute slope over previous hedge_lookback bars
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')

    for i in range(hedge_lookback, n + 1):
        # Window is [i - hedge_lookback, i)
        start = i - hedge_lookback
        end = i
        x = close_b[start:end]
        y = close_a[start:end]

        # Skip window if any NaNs present
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        # Skip if x has zero variance (would lead to invalid slope)
        if np.allclose(x, x[0]):
            continue

        # Compute OLS slope
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[end - 1] = float(slope)
        except Exception:
            # Regression failed for this window; leave as NaN
            hedge_ratio[end - 1] = np.nan

    # Compute spread using the rolling hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Compute rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Avoid division by zero
    spread_std_safe = np.where(spread_std == 0, np.nan, spread_std)

    zscore = (spread - spread_mean) / spread_std_safe

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
