"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators: compute hedge ratio (rolling OLS), spread, and z-score
- order_func: order function for vectorbt flexible multi-asset simulation

Notes:
- No numba usage.
- Returns plain Python tuples from order_func.
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
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices of asset A (array-like)
        close_b: Prices of asset B (array-like)
        hedge_lookback: Lookback window for rolling OLS (maximum window). Implementation
                        uses a growing window for early bars (uses all available history up
                        to `hedge_lookback` to avoid NaNs).
        zscore_lookback: Lookback window for rolling mean/std of spread. Uses growing
                         window behavior for early bars.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of slope coefficients (length == len(close_a))
            - 'spread': np.ndarray of spread values
            - 'zscore': np.ndarray of z-score values

    Behavior notes (important for backtest invariants):
        - No lookahead: values at time t only depend on data <= t
        - Avoids NaNs by using growing windows (so early bars produce valid numbers)
    """
    # Convert inputs to numpy arrays
    pa = np.asarray(close_a, dtype=float)
    pb = np.asarray(close_b, dtype=float)

    if pa.shape != pb.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = pa.shape[0]

    hedge_ratio = np.empty(n, dtype=float)
    spread = np.empty(n, dtype=float)
    zscore = np.empty(n, dtype=float)

    prev_slope = 1.0  # fallback initial slope

    # Helper epsilon
    EPS = 1e-8

    for t in range(n):
        # Rolling window indices (inclusive)
        start = 0 if t - hedge_lookback + 1 < 0 else t - hedge_lookback + 1

        x = pb[start : t + 1]
        y = pa[start : t + 1]

        # If insufficient variance in x, keep previous slope
        if x.size < 2:
            slope = prev_slope
        else:
            xm = x.mean()
            ym = y.mean()
            denom = np.sum((x - xm) ** 2)
            if denom > EPS:
                cov = np.sum((x - xm) * (y - ym))
                slope = cov / denom
            else:
                slope = prev_slope

        hedge_ratio[t] = slope
        prev_slope = slope

        # Spread at time t (no lookahead)
        spread[t] = pa[t] - slope * pb[t]

        # Rolling mean/std of spread using growing window up to zscore_lookback
        zs_start = 0 if t - zscore_lookback + 1 < 0 else t - zscore_lookback + 1
        window = spread[zs_start : t + 1]
        m = window.mean()
        s = window.std(ddof=0)  # population std to avoid NaN for single value
        if s < EPS:
            zscore[t] = 0.0
        else:
            zscore[t] = (spread[t] - m) / s

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
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset vectorbt backtest.

    This function is written to be called for each column (asset) separately in a
    flexible mode wrapper. It returns a tuple (size, size_type, direction):
        - size: positive float amount (in units when size_type == 0)
        - size_type: 0 -> Amount (units)
        - direction: 0 -> LongOnly, 1 -> ShortOnly, 2 -> Both

    Logic implemented:
        - Entry when zscore > entry_threshold -> SHORT A, LONG B
        - Entry when zscore < -entry_threshold -> LONG A, SHORT B
        - Exit when zscore crosses zero (sign change) or |zscore| > stop_threshold
        - Position sizing: fixed notional per leg -> units = notional / price
          Asset B size is scaled by hedge_ratio (units_b = units_a * hedge_ratio)

    Notes:
        - No numba usage. Returns simple Python tuple. To signal NoOrder, returns
          (np.nan, 0, 0) and the caller/wrapper treats NaN size as no order.
        - Uses only current and past data (c.i and zscore indices) -> no lookahead.
    """
    # Constants for size_type and direction (avoid importing vbt.enums in user code)
    SIZE_TYPE_AMOUNT = 0
    SIZE_TYPE_TARGET_AMOUNT = 3
    DIR_LONG = 0
    DIR_SHORT = 1
    DIR_BOTH = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive conversions
    pa = np.asarray(close_a, dtype=float)
    pb = np.asarray(close_b, dtype=float)
    zs = np.asarray(zscore, dtype=float)
    hr = np.asarray(hedge_ratio, dtype=float)

    # Guard for index range
    if i < 0 or i >= zs.shape[0]:
        return (np.nan, SIZE_TYPE_AMOUNT, DIR_LONG)

    # Current values
    z = float(zs[i]) if not np.isnan(zs[i]) else 0.0
    hedge = float(hr[i]) if not np.isnan(hr[i]) else 0.0

    price_a = float(pa[i]) if not np.isnan(pa[i]) else np.nan
    price_b = float(pb[i]) if not np.isnan(pb[i]) else np.nan

    # Get current position for this column (units). If missing, assume flat.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Compute intended unit sizes for new entries
    units_a = 0.0
    units_b = 0.0
    if not np.isnan(price_a) and price_a > 0:
        units_a = float(notional_per_leg) / price_a
    if not np.isnan(price_b) and price_b > 0:
        # size for B is based on units_a scaled by hedge ratio
        units_b = units_a * hedge

    # Entry signals (based only on current z)
    enter_short_pair = z > entry_threshold
    enter_long_pair = z < -entry_threshold

    # Stop loss
    stop_loss = abs(z) > stop_threshold

    # Zero-cross detection: compare with previous z (no lookahead)
    cross_zero = False
    if i > 0:
        prev_z = float(zs[i - 1]) if not np.isnan(zs[i - 1]) else 0.0
        # Consider crossing if signs are opposite
        if prev_z * z < 0:
            cross_zero = True
        # Also consider exact zero
        if z == 0.0 and prev_z != 0.0:
            cross_zero = True

    exit_pair = cross_zero or stop_loss

    # If flat -> consider entry
    if abs(pos_now) < 1e-12:
        if enter_short_pair:
            # Short A, Long B
            if col == 0:
                # Asset A: short
                size = units_a
                return (size, SIZE_TYPE_AMOUNT, DIR_SHORT)
            else:
                # Asset B: long
                size = units_b
                return (size, SIZE_TYPE_AMOUNT, DIR_LONG)

        if enter_long_pair:
            # Long A, Short B
            if col == 0:
                # Asset A: long
                size = units_a
                return (size, SIZE_TYPE_AMOUNT, DIR_LONG)
            else:
                # Asset B: short
                size = units_b
                return (size, SIZE_TYPE_AMOUNT, DIR_SHORT)

        # No entry
        return (np.nan, SIZE_TYPE_AMOUNT, DIR_LONG)

    # If in a position -> consider exit
    if abs(pos_now) > 1e-12 and exit_pair:
        # Use target amount = 0 to close position cleanly (direction Both)
        return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIR_BOTH)

    # Otherwise, do nothing
    return (np.nan, SIZE_TYPE_AMOUNT, DIR_LONG)
