"""
Pairs trading strategy utilities

Exports:
- compute_spread_indicators
- order_func

Implements a rolling OLS hedge ratio, spread, and z-score computation and a
flexible order function for vectorbt's flexible order mode.

Notes:
- Uses expanding windows when not enough history is available (no lookahead).
- Avoids NaNs by falling back to sensible defaults (slope=0, zscore=0) when
  statistics are degenerate.
- Order function returns simple tuples (size, size_type, direction) as
  required by the test harness wrapper which converts them to vectorbt orders.

"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Constants for vectorbt order size types and directions.
# These integer values correspond to vectorbt.portfolio.enums.SizeTypeT and
# vectorbt.portfolio.enums.DirectionT as documented:
# SizeTypeT(Amount=0, Value=1, Percent=2, TargetAmount=3, TargetValue=4, TargetPercent=5)
# DirectionT(LongOnly=0, ShortOnly=1, Both=2)
SIZE_TYPE_AMOUNT = 0
SIZE_TYPE_VALUE = 1
SIZE_TYPE_PERCENT = 2
SIZE_TYPE_TARGET_AMOUNT = 3
SIZE_TYPE_TARGET_VALUE = 4
SIZE_TYPE_TARGET_PERCENT = 5

DIRECTION_LONG = 0   # LongOnly
DIRECTION_SHORT = 1  # ShortOnly
DIRECTION_BOTH = 2   # Both


def compute_spread_indicators(
    close_a: np.ndarray | pd.Series,
    close_b: np.ndarray | pd.Series,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS), spread and z-score.

    Args:
        close_a: Prices of asset A (array-like)
        close_b: Prices of asset B (array-like)
        hedge_lookback: Lookback for rolling OLS (uses up to this many past
            observations, but will use fewer if not enough history is available)
        zscore_lookback: Lookback for rolling mean/std of spread used to compute
            z-score (uses up to this many past observations)

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'spread': np.ndarray of spread values
            - 'zscore': np.ndarray of z-score values

    Implementation notes:
    - Uses only information up to and including the current index (no lookahead).
    - When variance of the regressor is zero, falls back to hedge_ratio=0.0.
    - When std of spread is zero, z-score is set to 0.0.
    """
    # Convert inputs to numpy arrays of float
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Rolling OLS for hedge ratio: regress A on B -> A = slope * B + eps
    # For each time t use data in window [t - hedge_lookback + 1, t], but if
    # not enough data is available, use all data up to t (expanding behavior).
    eps = 1e-12
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        a_w = a[start : i + 1]
        b_w = b[start : i + 1]

        # If all values are nan or window empty, fallback
        if a_w.size == 0 or b_w.size == 0:
            slope = 0.0
        else:
            # Compute slope = cov(a,b)/var(b)
            b_mean = np.nanmean(b_w)
            a_mean = np.nanmean(a_w)
            denom = np.nansum((b_w - b_mean) ** 2)
            if denom <= eps:
                slope = 0.0
            else:
                cov = np.nansum((b_w - b_mean) * (a_w - a_mean))
                slope = cov / denom

        hedge_ratio[i] = slope
        spread[i] = a[i] - slope * b[i]

    # Rolling mean/std for spread (z-score)
    for i in range(n):
        start = max(0, i - zscore_lookback + 1)
        s_w = spread[start : i + 1]
        if s_w.size == 0:
            mu = 0.0
            sigma = 0.0
        else:
            mu = np.nanmean(s_w)
            sigma = float(np.nanstd(s_w, ddof=0))

        if sigma <= eps:
            z = 0.0
        else:
            z = (spread[i] - mu) / sigma

        zscore[i] = z

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """Order function for pairs trading in flexible multi-asset mode.

    This function returns a simple tuple (size, size_type, direction) which
    the harness wrapper converts to vectorbt orders. It is designed to be
    called for each asset (col) separately at each bar.

    Logic:
    - Entry when z-score > entry_threshold or z-score < -entry_threshold.
    - Exit when z-score crosses zero (mean reversion) or when |z-score| > stop_threshold (stop-loss).
    - Position sizing:
        * Base units for Asset A = notional_per_leg / price_A
        * Asset B units = abs(hedge_ratio) * base_units
      Orders are issued as TARGET_AMOUNT to avoid repeated accumulation while
      the signal persists.

    Returns:
        (size, size_type, direction)
        - For no order: return (np.nan, 0, 0)
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive indexing
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Avoid trading on non-positive prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    h = float(hedge_ratio[i]) if i < len(hedge_ratio) else 0.0

    # Current position (signed amount, positive = long, negative = short)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Compute base unit size for asset A (units) based on notional per leg
    base_units_a = float(notional_per_leg) / price_a if price_a > 0 else 0.0
    # Asset B units are scaled by hedge ratio magnitude
    base_units_b = abs(h) * base_units_a

    # Small tolerance for comparing current position to desired
    tol = 1e-8

    # 1) Stop-loss: close both legs if |z| > stop_threshold
    if np.isfinite(z) and abs(z) > stop_threshold:
        # If we already have no position, do nothing
        if abs(pos_now) <= tol:
            return (np.nan, 0, 0)
        # Set target amount to zero (close position). Use BOTH direction to
        # ensure both long/short are closed.
        return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # 2) Mean-reversion exit: z-score crosses zero -> close positions
    if i > 0:
        z_prev = float(zscore[i - 1])
        crossed_zero = ((z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0))
        if crossed_zero:
            if abs(pos_now) <= tol:
                return (np.nan, 0, 0)
            return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # 3) Entry conditions
    # Long A / Short B when z < -entry_threshold
    if np.isfinite(z) and z < -float(entry_threshold):
        if col == 0:
            # Asset A: go LONG base_units_a
            desired_size = base_units_a
            desired_dir = DIRECTION_LONG
            desired_signed = desired_size
        else:
            # Asset B: go SHORT base_units_b
            desired_size = base_units_b
            desired_dir = DIRECTION_SHORT
            desired_signed = -desired_size

        # Check if we already have the desired position
        if desired_dir == DIRECTION_LONG:
            if pos_now > 0 and abs(pos_now - desired_signed) <= max(tol, abs(desired_signed) * 1e-6):
                return (np.nan, 0, 0)
        elif desired_dir == DIRECTION_SHORT:
            if pos_now < 0 and abs(abs(pos_now) - abs(desired_signed)) <= max(tol, abs(desired_signed) * 1e-6):
                return (np.nan, 0, 0)

        # Issue target amount order
        return (float(desired_size), SIZE_TYPE_TARGET_AMOUNT, int(desired_dir))

    # Short A / Long B when z > entry_threshold
    if np.isfinite(z) and z > float(entry_threshold):
        if col == 0:
            # Asset A: go SHORT base_units_a
            desired_size = base_units_a
            desired_dir = DIRECTION_SHORT
            desired_signed = -desired_size
        else:
            # Asset B: go LONG base_units_b
            desired_size = base_units_b
            desired_dir = DIRECTION_LONG
            desired_signed = desired_size

        # Check if we already have the desired position
        if desired_dir == DIRECTION_LONG:
            if pos_now > 0 and abs(pos_now - desired_signed) <= max(tol, abs(desired_signed) * 1e-6):
                return (np.nan, 0, 0)
        elif desired_dir == DIRECTION_SHORT:
            if pos_now < 0 and abs(abs(pos_now) - abs(desired_signed)) <= max(tol, abs(desired_signed) * 1e-6):
                return (np.nan, 0, 0)

        return (float(desired_size), SIZE_TYPE_TARGET_AMOUNT, int(desired_dir))

    # No order by default
    return (np.nan, 0, 0)
