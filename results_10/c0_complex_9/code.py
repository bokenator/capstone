"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Hedge ratio is estimated with rolling OLS (slope) using scipy.stats.linregress.
- Z-score computed from rolling mean/std of the spread.
- Entry/exit/stop rules implemented as per specification.

CRITICAL: Does not use numba.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

# We import enums dynamically inside order_func to be resilient to enum naming
# and avoid tight coupling at module import time.


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS slope) and spread z-score for a pair of assets.

    Args:
        close_a: Close prices for asset A (1D numpy array).
        close_b: Close prices for asset B (1D numpy array).
        hedge_lookback: Lookback window (in bars) for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread used to compute z-score.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": numpy array of same length as inputs with rolling OLS slopes.
            - "zscore": numpy array of z-score of the spread.

    Notes:
        - Regions with insufficient data or NaNs will contain np.nan.
        - Uses scipy.stats.linregress for each rolling window (non-numba, Python loop).
    """
    # Validate inputs
    if not isinstance(close_a, np.ndarray):
        close_a = np.asarray(close_a)
    if not isinstance(close_b, np.ndarray):
        close_b = np.asarray(close_b)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: estimate slope of close_a ~ slope * close_b + intercept
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    # Compute slopes for each window end at i (inclusive)
    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        y = close_a[start : i + 1]
        x = close_b[start : i + 1]

        # Require full window and finite values
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        # Avoid degenerate regression when x is constant
        if np.allclose(x, x[0]):
            # If x is constant, slope is undefined -> keep NaN
            continue

        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            # Accept slope as hedge ratio (could be zero in degenerate cases)
            if np.isfinite(slope):
                hedge_ratio[i] = float(slope)
        except Exception:
            # Any failure leaves hedge_ratio as NaN
            continue

    # Compute spread using hedge ratio; where hedge_ratio is NaN -> spread NaN
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = ~np.isnan(hedge_ratio)
    if valid_idx.any():
        spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Compute rolling mean and std for z-score using pandas for convenience
    spread_s = pd.Series(spread)
    # Use min_periods=zscore_lookback to require full window
    roll_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to keep z-score stable for small samples; sample std (ddof=1) could be used as well
    roll_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    zscore_s = (spread_s - roll_mean) / roll_std

    # Convert results to numpy arrays
    result = {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore_s.to_numpy(dtype=float),
    }

    return result



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
    Order function for flexible multi-asset vectorbt execution.

    This function is called once per asset (column) per bar in flexible mode. It returns a tuple
    (size, size_type, direction). To indicate no order, return a tuple with size = np.nan.

    Logic implemented:
    - Entry: z-score > entry_threshold => Short A, Long B
             z-score < -entry_threshold => Long A, Short B
      Sizes:
        - size_A = notional_per_leg / price_A (units)
        - size_B = hedge_ratio * (notional_per_leg / price_B)
      (This keeps the notional per A leg at the specified value and scales B by hedge_ratio.)

    - Exit: z-score crosses zero => close both legs
    - Stop-loss: |z-score| > stop_threshold => close both legs

    Notes:
    - Uses the provided c object for context: c.i (bar index), c.col (column index), c.position_now (current position units)
    - Returns simple Python tuple, without numba objects.
    """
    # Extract index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Current position for this column (in units/shares)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Basic sanity checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = zscore[i]
    hr = hedge_ratio[i]

    # If indicators are not available or prices non-positive -> no action
    if not np.isfinite(z) or not np.isfinite(hr) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine appropriate enum values for size_type and directions in a robust way
    try:
        from vectorbt.portfolio.enums import SizeType, Direction

        # Choose a member of SizeType that represents "size in shares/units".
        size_type_member = None
        for m in SizeType:
            mname = m.name.lower()
            if "size" in mname or "share" in mname or "absolute" in mname or "volume" in mname:
                size_type_member = m
                break
        if size_type_member is None:
            size_type_member = list(SizeType)[0]
        size_type_int = int(size_type_member)

        # Direction members
        dir_long = None
        dir_short = None
        for m in Direction:
            mname = m.name.lower()
            if mname.startswith("long"):
                dir_long = m
            if mname.startswith("short"):
                dir_short = m
        # Fallbacks
        dirs = list(Direction)
        if dir_long is None:
            dir_long = dirs[0]
        if dir_short is None:
            dir_short = dirs[1] if len(dirs) > 1 else dirs[0]

        dir_long_int = int(dir_long)
        dir_short_int = int(dir_short)
    except Exception:
        # If enums are not available for some reason, fall back to safe integers.
        # These integer choices are generic and may map in vectorbt to the expected actions.
        size_type_int = 1
        dir_long_int = 1
        dir_short_int = 2

    # Compute sizes in units (shares)
    size_a = float(notional_per_leg / price_a) if price_a > 0 else 0.0
    # Scale B by hedge ratio so that B's units reflect the hedge relationship
    size_b = float(hr * (notional_per_leg / price_b)) if price_b > 0 else 0.0

    # Helper to emit a close order for the current position
    def close_order_for_position(position_units: float) -> Tuple[float, int, int]:
        if position_units == 0:
            return (np.nan, 0, 0)
        # To close a long position (positive), issue a SHORT (sell) order of that many units
        # To close a short position (negative), issue a LONG (buy) order of abs(units)
        if position_units > 0:
            return (float(abs(position_units)), size_type_int, dir_short_int)
        else:
            return (float(abs(position_units)), size_type_int, dir_long_int)

    # 1) Stop-loss has highest priority: if |z| > stop_threshold -> close any open position
    if np.isfinite(stop_threshold) and abs(z) > float(stop_threshold):
        return close_order_for_position(pos_now)

    # 2) Exit on crossing the exit_threshold (typically 0.0)
    crossed = False
    if i > 0:
        prev_z = zscore[i - 1]
        if np.isfinite(prev_z) and np.isfinite(z):
            # Detect sign crossing through the threshold (exit_threshold)
            # We implement crossing of zero specifically since exit_threshold is provided as 0.0 in the backtest.
            # More generally, detect sign change relative to exit_threshold.
            thr = float(exit_threshold)
            prev_rel = prev_z - thr
            curr_rel = z - thr
            if prev_rel * curr_rel < 0:
                crossed = True
            elif prev_rel != 0 and curr_rel == 0:
                crossed = True

    if crossed:
        return close_order_for_position(pos_now)

    # 3) Entry logic: only enter if there is no existing position for this column
    #    z > entry_threshold => Short A, Long B
    if z > float(entry_threshold):
        if col == 0:
            # Asset A: short
            if pos_now == 0:
                return (float(size_a), size_type_int, dir_short_int)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: long
            if pos_now == 0:
                return (float(size_b), size_type_int, dir_long_int)
            else:
                return (np.nan, 0, 0)

    #    z < -entry_threshold => Long A, Short B
    if z < -float(entry_threshold):
        if col == 0:
            # Asset A: long
            if pos_now == 0:
                return (float(size_a), size_type_int, dir_long_int)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: short
            if pos_now == 0:
                return (float(abs(size_b)), size_type_int, dir_short_int)
            else:
                return (np.nan, 0, 0)

    # Default: no order
    return (np.nan, 0, 0)
