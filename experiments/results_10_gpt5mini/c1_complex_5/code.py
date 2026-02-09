import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple, Any


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
) -> Tuple[float, int, int]:
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

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    """
    # Extract context
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic guards
    # Ensure index is within arrays
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # If zscore is not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio
    h = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if np.isnan(h):
        return (np.nan, 0, 0)

    # Compute share sizes (can be fractional)
    try:
        shares_a = float(notional_per_leg) / float(price_a)
    except Exception:
        return (np.nan, 0, 0)

    try:
        # Multiply by hedge ratio so we trade hedge_ratio units of B per the strategy
        shares_b = (float(notional_per_leg) / float(price_b)) * float(h)
    except Exception:
        return (np.nan, 0, 0)

    # Determine exit (crossing zero) using previous z-score when available
    z_prev = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if not np.isnan(z_prev):
        # Crossing if signs differ or one is exactly zero
        if z_prev * z <= 0:
            crossed_zero = True

    # Also consider exit when within a small threshold band if exit_threshold > 0
    within_exit_band = False
    if exit_threshold is not None and exit_threshold > 0 and abs(z) <= exit_threshold:
        within_exit_band = True

    # Stop-loss
    stop_loss = abs(z) > stop_threshold

    # Determine desired positions for both assets
    # None -> no order (keep current), 0.0 -> flat, nonzero -> target shares
    desired_a = None
    desired_b = None

    if stop_loss or crossed_zero or within_exit_band:
        # Close both legs
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entries
        if z > entry_threshold:
            # Short Asset A, Long Asset B (hedge_ratio units)
            desired_a = -shares_a
            desired_b = +shares_b
        elif z < -entry_threshold:
            # Long Asset A, Short Asset B
            desired_a = +shares_a
            desired_b = -shares_b
        else:
            # No trading signal; do nothing
            return (np.nan, 0, 0)

    # Choose which asset we're placing order for
    if col == 0:
        target = desired_a
    else:
        target = desired_b

    # If for some reason target is None, do nothing
    if target is None:
        return (np.nan, 0, 0)

    # Compute delta (order size = desired - current)
    delta = float(target) - float(pos)

    # If delta is effectively zero, no action
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return order: number of shares (Amount), allow both long and short
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible: it accepts either pandas DataFrame/Series with a
    'close' column or a 1-D numpy array of close prices for each asset. It
    returns numpy arrays for close prices, rolling hedge ratio (OLS slope),
    and z-score of the spread.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B or numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close arrays from inputs (support DataFrame or ndarray)
    def _extract_close(x: Any) -> np.ndarray:
        # If it's a DataFrame, extract 'close' column
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            arr = x['close'].values.astype(float)
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Try to coerce
            arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            # If a 2-D array with one column, flatten
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError('Close price input must be 1-D')
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 1 or hedge_lookback > n:
        raise ValueError('hedge_lookback must be >=1 and <= length of data')
    if zscore_lookback < 1 or zscore_lookback > n:
        raise ValueError('zscore_lookback must be >=1 and <= length of data')

    # Rolling hedge ratio (OLS slope) -- allocate array of NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # For each window ending at index i (exclusive), compute slope using previous
    # hedge_lookback points: slice [i-hedge_lookback:i]
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Skip if NaNs in window
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x is constant, slope is undefined -> set NaN
        if np.allclose(x, x[0]):
            hedge_ratio[i] = np.nan
            continue

        # Compute OLS slope y = slope * x + intercept
        try:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge ratio (element-wise). If hedge_ratio is NaN -> spread NaN
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std of spread
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - spread_mean[valid_mask]) / spread_std[valid_mask]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
