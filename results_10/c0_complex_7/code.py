"""
Pairs trading strategy implementation for vectorbt backtest runner.

Exports:
- compute_spread_indicators
- order_func

Notes:
- No numba used.
- Does not use vbt.portfolio.enums.

Strategy logic:
- Hedge ratio: rolling OLS (lookback, default 60)
- Spread = A - hedge_ratio * B
- Z-score: rolling mean/std of spread (lookback, default 20)
- Entry: zscore > entry_threshold -> short A, long B
         zscore < -entry_threshold -> long A, short B
- Exit: zscore crosses zero OR |zscore| > stop_threshold -> close positions
- Position sizing: fixed notional per leg (notional_per_leg)

Return conventions for order_func:
- Returns (size, size_type, direction)
- size: number of units to trade (absolute)
- size_type: 0 (absolute number of units)
- direction: 1 -> BUY (go/close long), 2 -> SELL (go/close short)
- To indicate no order return (np.nan, 0, 0)

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
    Compute rolling hedge ratio (OLS), spread, rolling mean/std and z-score.

    Args:
        close_a: prices for asset A (1D array)
        close_b: prices for asset B (1D array)
        hedge_lookback: lookback for rolling OLS (integer)
        zscore_lookback: lookback for rolling mean/std of spread (integer)

    Returns:
        dict with keys: 'hedge_ratio', 'spread', 'mean', 'std', 'zscore'
        Each value is a numpy array of same length as inputs (NaNs for warmup).
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: slope b = cov(x, y) / var(x) where we regress A on B (A = alpha + beta * B)
    for t in range(hedge_lookback - 1, n):
        start = t - hedge_lookback + 1
        win_a = a[start : t + 1]
        win_b = b[start : t + 1]

        # Skip windows with NaNs
        if np.isnan(win_a).any() or np.isnan(win_b).any():
            continue

        mean_b = win_b.mean()
        mean_a = win_a.mean()
        cov = np.sum((win_b - mean_b) * (win_a - mean_a))
        var_b = np.sum((win_b - mean_b) ** 2)

        if var_b == 0:
            hedge_ratio[t] = np.nan
        else:
            hedge_ratio[t] = cov / var_b

    # Spread: A - hedge_ratio * B
    spread = a - hedge_ratio * b

    # Rolling mean/std (use pandas for convenience and NaN handling)
    s = pd.Series(spread)
    mean_spread = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to avoid NaNs for small windows once min_periods met
    std_spread = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score, guard against division by zero
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(mean_spread)) & (~np.isnan(std_spread)) & (std_spread != 0)
    zscore[valid] = (spread[valid] - mean_spread[valid]) / std_spread[valid]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "mean": mean_spread,
        "std": std_spread,
        "zscore": zscore,
    }


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
    Order function for flexible (multi-asset) vectorbt backtest.

    This function is designed to be called for each asset (col) on each bar.
    The wrapper will call this function per column and aggregate orders.

    Returns a tuple (size, size_type, direction):
      - size: absolute number of units to trade (float)
      - size_type: 0 -> absolute units
      - direction: 1 -> BUY (long), 2 -> SELL (short)

    To indicate no order, return (np.nan, 0, 0).
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Constants for size_type and directions
    SIZE_TYPE_ABSOLUTE = 0
    DIRECTION_BUY = 1
    DIRECTION_SELL = 2

    # Basic guards
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # If price invalid, skip
    if price <= 0 or np.isnan(price):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan

    # If zscore not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Get previous zscore for crossing detection
    prev_z = np.nan
    if i > 0:
        prev_z = zscore[i - 1]

    # Current position (units). It might be array-like; ensure scalar
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(np.asarray(pos_now).item())
    except Exception:
        # Fallback: if it's not convertible, treat as 0
        try:
            pos_now = float(pos_now)
        except Exception:
            pos_now = 0.0

    # Helper: check if position is effectively zero
    def is_zero(x: float, tol: float = 1e-8) -> bool:
        return abs(x) <= tol

    # Exit conditions: stop-loss or z-score crossing zero
    exit_signal = False
    # Stop-loss
    if abs(z) > float(stop_threshold):
        exit_signal = True

    # Cross zero detection: prev_z * z < 0 (and prev_z not NaN)
    if not np.isnan(prev_z) and (prev_z * z) < 0:
        exit_signal = True

    if exit_signal:
        # If we have a position, close it fully
        if not is_zero(pos_now):
            size = abs(pos_now)
            if pos_now > 0:
                # currently long -> sell to close
                return (size, SIZE_TYPE_ABSOLUTE, DIRECTION_SELL)
            else:
                # currently short -> buy to close
                return (size, SIZE_TYPE_ABSOLUTE, DIRECTION_BUY)
        else:
            return (np.nan, 0, 0)

    # Entry signals
    # Only enter if we currently have no position in this asset
    if not is_zero(pos_now):
        # Already in a position for this asset; do nothing
        return (np.nan, 0, 0)

    # Long/short sizing based on fixed notional per leg
    units = notional_per_leg / price if notional_per_leg is not None else 0.0

    # Entry: z > entry_threshold -> SHORT A, LONG B
    if z > float(entry_threshold):
        if col == 0:
            # Asset A: short
            return (units, SIZE_TYPE_ABSOLUTE, DIRECTION_SELL)
        else:
            # Asset B: long
            return (units, SIZE_TYPE_ABSOLUTE, DIRECTION_BUY)

    # Entry: z < -entry_threshold -> LONG A, SHORT B
    if z < -float(entry_threshold):
        if col == 0:
            # Asset A: long
            return (units, SIZE_TYPE_ABSOLUTE, DIRECTION_BUY)
        else:
            # Asset B: short
            return (units, SIZE_TYPE_ABSOLUTE, DIRECTION_SELL)

    # Otherwise no order
    return (np.nan, 0, 0)
