# Pairs trading strategy utilities for vectorbt
# compute_spread_indicators and order_func implementations

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series, list],
    close_b: Union[np.ndarray, pd.Series, list],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS through origin), spread and z-score for a pairs strategy.

    Hedge ratio is computed with a rolling OLS regression through the origin (no intercept):
        hedge_ratio = sum(A * B) / sum(B * B)
    The window used is up to `hedge_lookback` but falls back to a smaller window at the start
    (i.e., uses available data rather than returning NaN). This avoids lookahead bias while
    ensuring outputs are defined early.

    Rolling mean and std for z-score use a window of `zscore_lookback` periods (also with
    fallback to available data at the beginning). If rolling std is extremely small, z-score
    is set to 0 to avoid divide-by-zero.

    Args:
        close_a: Price series for asset A (array-like)
        close_b: Price series for asset B (array-like)
        hedge_lookback: Lookback for rolling hedge ratio (default 60)
        zscore_lookback: Lookback for rolling mean/std of spread (default 20)

    Returns:
        dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as input)
            - 'spread': np.ndarray of spreads
            - 'zscore': np.ndarray of z-scores

    Notes:
        - The function is careful to avoid lookahead: values at index t are computed using
          data up to and including t only.
        - Outputs are numpy arrays of dtype float.
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Small epsilon for numeric stability
    EPS = 1e-8

    # Compute rolling hedge ratio (regression through origin) with expanding window up to hedge_lookback
    for t in range(n):
        end = t + 1
        win = min(hedge_lookback, end)
        start = end - win
        Awin = a[start:end]
        Bwin = b[start:end]

        # Skip NaNs in the window
        valid_mask = (~np.isnan(Awin)) & (~np.isnan(Bwin))
        if valid_mask.sum() == 0:
            # no valid data
            hedge_ratio[t] = np.nan
            continue

        Awin_v = Awin[valid_mask]
        Bwin_v = Bwin[valid_mask]

        denom = np.dot(Bwin_v, Bwin_v)
        if denom > EPS:
            hedge_ratio[t] = float(np.dot(Awin_v, Bwin_v) / denom)
        else:
            # If denominator too small, fallback to previous hedge_ratio or 0
            hedge_ratio[t] = hedge_ratio[t - 1] if t > 0 and not np.isnan(hedge_ratio[t - 1]) else 0.0

        # Compute spread at t
        spread[t] = a[t] - hedge_ratio[t] * b[t]

    # Rolling mean and std of spread for z-score using window zscore_lookback (fallback to available data)
    for t in range(n):
        end = t + 1
        win = min(zscore_lookback, end)
        start = end - win
        vals = spread[start:end]

        # Exclude NaNs
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            zscore[t] = np.nan
            continue

        m = float(vals.mean())
        s = float(vals.std(ddof=0))  # population std
        if s < EPS:
            # Avoid division by zero; treat z-score as 0 when volatility extremely small
            zscore[t] = 0.0
        else:
            zscore[t] = float((spread[t] - m) / s)

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
    Order function for a pairs trading strategy (flexible multi-asset mode).

    This function is designed to be called once per asset per bar. It returns a tuple
    (size, size_type, direction). To indicate "no order", it returns (np.nan, 0, 0).

    Strategy rules implemented:
        - Entry:
            * zscore > entry_threshold: Short A, Long B
            * zscore < -entry_threshold: Long A, Short B
          Position sizing (units):
            * units_A = notional_per_leg / price_A
            * units_B = hedge_ratio * units_A
          Orders are submitted in Amount units (size_type = 0).

        - Exit:
            * zscore crosses 0.0 (sign change) -> close both positions
            * |zscore| > stop_threshold -> stop-loss, close both positions
          Closing is executed by submitting a TargetAmount order with target 0 (size_type = 3,
          direction = Both) which is direction-agnostic and closes any existing long/short.

    Notes:
        - c is the order context and must provide attributes `.i` (int index) and
          `.col` (int column) and `.position_now` (current position amount for the column).
        - The function does not rely on any future data: only zscore[i] and zscore[i-1]
          are accessed for crossing detection.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safety checks and extraction of current values
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    h = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Use enum-like integer codes to avoid importing vectorbt enums directly
    SIZE_TYPE_AMOUNT = 0  # Amount (units)
    SIZE_TYPE_TARGET_AMOUNT = 3  # Target amount
    DIRECTION_LONGONLY = 0
    DIRECTION_SHORTONLY = 1
    DIRECTION_BOTH = 2

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(h) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine previous zscore for zero-cross detection
    z_prev = None
    if i > 0 and not np.isnan(zscore[i - 1]):
        z_prev = float(zscore[i - 1])

    # Helper to compute units for asset A and asset B
    units_a = notional_per_leg / price_a
    units_b = abs(h) * units_a  # scale by absolute hedge ratio for sizing in units

    # Exit conditions: close if z crosses zero or stop-loss
    exit_condition = False
    if z_prev is not None:
        # Crossing zero: previous positive and current <= exit_threshold, or previous negative and current >= exit_threshold
        if (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold):
            exit_condition = True

    if abs(z) > stop_threshold:
        exit_condition = True

    if exit_condition and abs(pos_now) > 0:
        # Close current position by setting target amount to 0 (direction-agnostic)
        return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # Entry conditions: only open when currently flat
    if abs(pos_now) < 1e-12:
        # Short A, Long B (z > entry_threshold)
        if z > entry_threshold:
            if col == 0:
                # Short asset A by units_a
                return (float(units_a), SIZE_TYPE_AMOUNT, DIRECTION_SHORTONLY)
            else:
                # Long asset B by units_b
                return (float(units_b), SIZE_TYPE_AMOUNT, DIRECTION_LONGONLY)

        # Long A, Short B (z < -entry_threshold)
        if z < -entry_threshold:
            if col == 0:
                # Long asset A
                return (float(units_a), SIZE_TYPE_AMOUNT, DIRECTION_LONGONLY)
            else:
                # Short asset B
                return (float(units_b), SIZE_TYPE_AMOUNT, DIRECTION_SHORTONLY)

    # Otherwise, no order
    return (np.nan, 0, 0)
