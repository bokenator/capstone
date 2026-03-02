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
        c: vectorbt OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: Close prices for Asset A (1D array)
        close_b: Close prices for Asset B (1D array)
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
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic bounds checks
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i >= n or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])

    # If zscore not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    h = float(hedge_ratio[i])

    # Protect against invalid prices or hedge ratios
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target shares based on notional per leg and hedge ratio
    # shares_a: number of shares for Asset A based on $ notional
    shares_a = notional_per_leg / price_a
    # For Asset B we scale by hedge ratio to obtain relative exposure
    # If hedge ratio is nan or not finite, we can't size B leg -> no action
    if not np.isfinite(h):
        shares_b = np.nan
    else:
        shares_b = (notional_per_leg / price_b) * h

    # Helper: get previous zscore for crossing detection
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit conditions (highest priority): stop-loss or crossing exit
    stop_loss = np.isfinite(z) and abs(z) > stop_threshold
    crossed = False
    if np.isfinite(z_prev):
        # crossing zero (or the provided exit_threshold)
        # We treat crossing as sign change around exit_threshold
        try:
            crossed = (z_prev - exit_threshold) * (z - exit_threshold) < 0
        except Exception:
            crossed = False

    # If stop-loss or crossing, close existing positions for this asset
    if stop_loss or crossed:
        # If there's an open position, close it
        if pos != 0.0:
            # Return an order to offset current position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry logic
    enter_short_a_long_b = z > entry_threshold
    enter_long_a_short_b = z < -entry_threshold

    # No entry if hedge ratio for B is invalid when we need B leg
    if (enter_short_a_long_b or enter_long_a_short_b) and not np.isfinite(h):
        return (np.nan, 0, 0)

    # Determine desired target positions (in shares) for each asset
    desired_target_a = None
    desired_target_b = None

    if enter_short_a_long_b:
        desired_target_a = -shares_a
        desired_target_b = +shares_b
    elif enter_long_a_short_b:
        desired_target_a = +shares_a
        desired_target_b = -shares_b

    # If there's no entry signal for this bar, do nothing
    if desired_target_a is None:
        return (np.nan, 0, 0)

    # For the current column, compute order size to move from current position to desired target
    if col == 0:
        # Asset A
        order_size = float(desired_target_a - pos)
        # If order_size is effectively zero, no action
        if abs(order_size) < 1e-8:
            return (np.nan, 0, 0)
        return (order_size, 0, 0)
    else:
        # Asset B
        order_size = float(desired_target_b - pos)
        if abs(order_size) < 1e-8:
            return (np.nan, 0, 0)
        return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR a 1D numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Allow flexibility: inputs may be DataFrames/Series with 'close' or raw numpy arrays
    def _extract_close(obj: Any) -> np.ndarray:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            # If DataFrame with 'close' column, use it
            if isinstance(obj, pd.DataFrame) and 'close' in obj.columns:
                return obj['close'].astype(float).values
            # If Series, take its values
            if isinstance(obj, pd.Series):
                return obj.astype(float).values
            # If DataFrame without 'close', try first column
            if isinstance(obj, pd.DataFrame) and obj.shape[1] >= 1:
                return obj.iloc[:, 0].astype(float).values
        # If it's a numpy array or list-like, convert
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 0:
                # scalar -> invalid
                raise ValueError("Input close prices must be 1D")
            return arr.flatten()
        except Exception as e:
            raise ValueError(f"Unsupported input type for close prices: {e}")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset A and Asset B must have the same length")

    n = len(close_a)

    # Prepare hedge_ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # If lookback is larger than available data, we won't compute any ratios
    if hedge_lookback <= 1:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Require at least 2 valid (non-NaN) points for regression
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid_mask) < 2:
            hedge_ratio[i] = np.nan
            continue

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # If x has zero variance, skip
        if np.nanstd(x_valid) == 0:
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio (elementwise)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    # Only compute spread where hedge_ratio is available and prices are finite
    mask = valid_hr & np.isfinite(close_a) & np.isfinite(close_b)
    spread[mask] = close_a[mask] - hedge_ratio[mask] * close_b[mask]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Compute z-score, guarding against zero std
    zscore = np.full(n, np.nan, dtype=float)
    valid_stat = np.isfinite(spread_mean.values) & np.isfinite(spread_std.values) & (spread_std.values > 0)
    zscore[valid_stat] = (spread[valid_stat] - spread_mean.values[valid_stat]) / spread_std.values[valid_stat]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }