import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Tuple, Dict


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

    Args:
        c: OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array for the spread
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        (size, size_type, direction)
    """
    # Extract context
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Read values for this bar
    z = zscore[i]
    hr = hedge_ratio[i]
    price_a = close_a[i]
    price_b = close_b[i]

    # Guard against invalid data
    if not np.isfinite(z) or not np.isfinite(hr) or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    # Prevent division by zero or negative prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine number of shares based on fixed notional per leg
    shares_a = float(notional_per_leg / price_a)
    # Follow provided formula: shares_b scaled by hedge ratio
    shares_b = float((notional_per_leg / price_b) * hr)

    # Determine previous z for crossing detection
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Desired positions (in shares) for each asset
    desired_a = None
    desired_b = None

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0

    # Exit: z-score crosses exit_threshold -> close both legs
    elif i > 0 and np.isfinite(prev_z):
        crossed_down = (prev_z > exit_threshold) and (z <= exit_threshold)
        crossed_up = (prev_z < exit_threshold) and (z >= exit_threshold)
        if crossed_down or crossed_up:
            desired_a = 0.0
            desired_b = 0.0

    # Entry logic (only if not already set by stop/exit)
    if desired_a is None:
        if z > entry_threshold:
            # Short A, Long B
            desired_a = -shares_a
            desired_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            desired_a = +shares_a
            desired_b = -shares_b

    # If strategy does not require action for this bar/col -> return No action
    if desired_a is None or desired_b is None:
        return (np.nan, 0, 0)

    # Compute order size = desired_position - current_position
    if col == 0:
        # Asset A
        size = float(desired_a - pos_now)
        return (size, 0, 0)
    elif col == 1:
        # Asset B
        size = float(desired_b - pos_now)
        return (size, 0, 0)

    # Unknown column
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames (with 'close' column) or numpy arrays for asset_a/asset_b.

    Returns a dict with numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    """
    # Support both DataFrame/Series inputs and raw numpy arrays
    def _extract_close(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.DataFrame):
                if 'close' not in x.columns:
                    raise KeyError("asset DataFrame must contain 'close' column")
                return x['close'].values.astype(float)
            else:
                # Series
                return x.values.astype(float)
        elif isinstance(x, np.ndarray):
            return x.astype(float)
        else:
            # try to coerce
            return np.asarray(x, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset close arrays must have the same length')

    n = len(close_a)

    # Prepare hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (simple linear regression slope) for each window end at i-1->i inclusive
    # We compute slope only when full window of finite values is available
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough valid points to run regression
            continue

        # Use only valid subset
        x_valid = x[mask]
        y_valid = y[mask]

        # If constant x or y, linregress may still return slope=nan; guard against degenerate cases
        try:
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x_valid, y_valid)
        except Exception:
            # If regression fails for any reason, skip
            continue

        # Only accept finite slope
        if np.isfinite(slope):
            hedge_ratio[i] = float(slope)

    # Compute spread = A - hedge_ratio * B (where hedge_ratio known)
    spread = np.full(n, np.nan, dtype=float)
    mask_hr = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[mask_hr] = close_a[mask_hr] - hedge_ratio[mask_hr] * close_b[mask_hr]

    # Rolling mean and std for spread
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score, guarding against zero std
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }