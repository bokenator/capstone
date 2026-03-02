"""
Pairs trading strategy utilities for vectorbt backtests.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (lookback default 60)
- Spread = A - hedge_ratio * B
- Z-score computed with rolling mean/std (default window 20)
- Entry: z > entry_threshold -> short A, long B
         z < -entry_threshold -> long A, short B
- Exit: z crosses 0 OR |z| > stop_threshold -> close both legs
- Position sizing: fixed notional per leg (notional_per_leg)

CRITICAL: This implementation does NOT use numba.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of price series.

    Args:
        close_a: Prices for asset A (1D array-like).
        close_b: Prices for asset B (1D array-like).
        hedge_lookback: Lookback window (in bars) for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std used in z-score.

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of same length as inputs with rolling slope (NaN for warmup).
            - "spread": np.ndarray of the spread A - hedge_ratio * B (NaN where hedge_ratio is NaN).
            - "zscore": np.ndarray of z-score computed from rolling mean/std of spread (NaN for warmup).

    Notes:
        - This function is robust to NaNs: if a regression window contains any NaN, the hedge_ratio
          for that bar will be NaN.
        - Uses numpy.polyfit for OLS on each rolling window.
    """
    # Convert inputs to 1D float arrays
    a = np.asarray(close_a, dtype=float).ravel()
    b = np.asarray(close_b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: y = slope * x + intercept where y = A, x = B
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        end = i + 1
        y = a[start:end]
        x = b[start:end]

        # Require full window without NaNs to compute regression
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # Perform simple OLS via polyfit (degree 1)
        try:
            # polyfit returns [slope, intercept] for deg=1
            slope, _ = np.polyfit(x, y, 1)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(hedge_ratio)
    spread[valid] = a[valid] - hedge_ratio[valid] * b[valid]

    # Rolling mean and std for z-score
    s = pd.Series(spread)

    # Use min_periods = zscore_lookback to avoid spurious values in warmup
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    roll_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to avoid NaN when window length equals lookback
    roll_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    roll_mean_arr = roll_mean.values
    roll_std_arr = roll_std.values

    zscore = np.full(n, np.nan, dtype=float)
    ok = (~np.isnan(spread)) & (~np.isnan(roll_mean_arr)) & (~np.isnan(roll_std_arr)) & (roll_std_arr > 0)
    zscore[ok] = (spread[ok] - roll_mean_arr[ok]) / roll_std_arr[ok]

    return {"hedge_ratio": hedge_ratio, "spread": spread, "zscore": zscore}


def _extract_scalar_position(position_now: Any, col: int) -> float:
    """
    Helper to extract a scalar position for the given column from various possible types.
    """
    if position_now is None:
        return 0.0

    # If it's a scalar already
    try:
        return float(position_now)
    except Exception:
        pass

    # If it's indexable (list/ndarray), try to extract by column
    try:
        return float(position_now[col])
    except Exception:
        try:
            arr = np.asarray(position_now, dtype=float)
            if arr.size > 0:
                # Fallback: sum positions (shouldn't normally happen)
                return float(np.nansum(arr))
        except Exception:
            return 0.0

    return 0.0


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset mode (pairs trading).

    Signature expected by the backtest runner:
        order_func(c, close_a, close_b, zscore, hedge_ratio,
                   entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

    Returns a tuple (size, size_type, direction):
        - size: positive float for order size (in units determined by size_type), or np.nan for NoOrder
        - size_type: integer (0 = amount (shares), 1 = value). We use 0 (amount)
        - direction: integer encoding order direction (1 = long/buy, 2 = short/sell). 0 indicates NoOrder.

    Rules implemented:
        - If zscore is NaN at the current bar -> NoOrder
        - If currently flat (position ~ 0): enter on zscore thresholds
            - z > entry_threshold: short A, long B
            - z < -entry_threshold: long A, short B
          Each leg sized as notional_per_leg / price (fixed-dollar per leg)
        - If currently in position: exit when zscore crosses zero (sign change from previous bar)
          OR when |zscore| > stop_threshold (stop-loss)

    Notes:
        - This function is stateless across calls; vectorbt's flexible wrapper will call it for each
          column separately for the same bar. The wrapper provides c.col to indicate which asset
          (0 -> asset_a, 1 -> asset_b) and c.position_now as the current position for that asset.
    """
    # Constants for size type and direction (use integers; vectorbt's wrapper will convert to enums)
    SIZE_TYPE_AMOUNT = 0
    DIR_LONG = 1
    DIR_SHORT = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Protect index bounds
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # If price is NaN or zero, do not place orders
    if not np.isfinite(price) or price <= 0:
        return (np.nan, 0, 0)

    # Current zscore and previous zscore (for crossing detection)
    z = float(zscore[i]) if (i >= 0 and np.isfinite(zscore[i])) else np.nan
    prev_z = float(zscore[i - 1]) if (i > 0 and np.isfinite(zscore[i - 1])) else np.nan

    # Current position for this column (in shares)
    raw_pos = getattr(c, "position_now", 0.0)
    pos_now = _extract_scalar_position(raw_pos, col)

    # Treat tiny positions as flat
    if np.isfinite(pos_now) and abs(pos_now) < 1e-8:
        pos_now = 0.0

    # If we don't have a valid zscore, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Helper to compute fixed-dollar size in units (shares)
    def dollar_size(p: float) -> float:
        return float(notional_per_leg) / float(p) if (p is not None and p > 0) else np.nan

    # ENTRY: only if currently flat
    if pos_now == 0.0:
        # Short A, Long B when zscore > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short asset A
                size = dollar_size(price)
                if not np.isfinite(size) or size <= 0:
                    return (np.nan, 0, 0)
                return (size, SIZE_TYPE_AMOUNT, DIR_SHORT)
            else:
                # Long asset B
                size = dollar_size(price)
                if not np.isfinite(size) or size <= 0:
                    return (np.nan, 0, 0)
                return (size, SIZE_TYPE_AMOUNT, DIR_LONG)

        # Long A, Short B when zscore < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long asset A
                size = dollar_size(price)
                if not np.isfinite(size) or size <= 0:
                    return (np.nan, 0, 0)
                return (size, SIZE_TYPE_AMOUNT, DIR_LONG)
            else:
                # Short asset B
                size = dollar_size(price)
                if not np.isfinite(size) or size <= 0:
                    return (np.nan, 0, 0)
                return (size, SIZE_TYPE_AMOUNT, DIR_SHORT)

        # No entry signal
        return (np.nan, 0, 0)

    # EXIT / STOP: if in position, close on z-score crossing zero or stop-loss
    crossed_zero = False
    if np.isfinite(prev_z):
        crossed_zero = (prev_z < 0 and z >= 0) or (prev_z > 0 and z <= 0)

    hit_stop = abs(z) > stop_threshold if np.isfinite(z) else False

    if crossed_zero or hit_stop:
        # Close this leg: determine direction based on current sign of position
        if pos_now > 0:
            # Currently long -> sell to close
            size = abs(pos_now)
            if size <= 0:
                return (np.nan, 0, 0)
            return (size, SIZE_TYPE_AMOUNT, DIR_SHORT)
        elif pos_now < 0:
            # Currently short -> buy to cover
            size = abs(pos_now)
            if size <= 0:
                return (np.nan, 0, 0)
            return (size, SIZE_TYPE_AMOUNT, DIR_LONG)

    # Otherwise do nothing
    return (np.nan, 0, 0)
