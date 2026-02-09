# Pairs trading strategy implementation for vectorbt
# Implements compute_spread_indicators and order_func as required by the backtest runner.

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS) and z-score of the spread.

    Args:
        close_a: 1D array of prices for asset A.
        close_b: 1D array of prices for asset B.
        hedge_lookback: Window length (in bars) for rolling OLS to compute hedge ratio.
        zscore_lookback: Window length for rolling mean/std of the spread.

    Returns:
        dict with keys:
            - "hedge_ratio": 1D array of hedge ratios (same length as inputs).
            - "zscore": 1D array of z-score values (same length as inputs).

    Notes:
        - Uses scipy.stats.linregress for OLS slope (hedge ratio).
        - Places NaN for indices where not enough history is available.
    """
    # Convert to numpy arrays and validate
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS to compute hedge_ratio (slope of regression of A on B)
    # We compute slope only when the window has no NaNs.
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        window_x = b[start_idx : end_idx + 1]
        window_y = a[start_idx : end_idx + 1]

        # Skip window if any NaNs present
        if np.isnan(window_x).any() or np.isnan(window_y).any():
            hedge_ratio[end_idx] = np.nan
            continue

        # If constant window (zero variance in x), slope is undefined
        if np.all(window_x == window_x[0]):
            hedge_ratio[end_idx] = np.nan
            continue

        # Compute OLS slope
        try:
            res = linregress(window_x, window_y)
            hedge_ratio[end_idx] = float(res.slope)
        except Exception:
            hedge_ratio[end_idx] = np.nan

    # Spread: A - hedge_ratio * B
    spread = a - hedge_ratio * b

    # Rolling mean and std for z-score (use pandas for convenience)
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    roll_mean_arr = roll_mean.to_numpy(dtype=float)
    roll_std_arr = roll_std.to_numpy(dtype=float)

    # z-score with safe division
    zscore = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(spread) & ~np.isnan(roll_mean_arr) & ~np.isnan(roll_std_arr) & (roll_std_arr > 0)
    zscore[valid] = (spread[valid] - roll_mean_arr[valid]) / roll_std_arr[valid]

    return {
        "hedge_ratio": hedge_ratio,
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
    """Order function for a flexible, two-asset pairs trading strategy.

    This function is called separately for each asset (col=0 for asset A, col=1 for asset B).
    It returns a tuple (size, size_type, direction) where:
      - size: positive float size in base asset units (shares). If NaN, no order is placed.
      - size_type: integer code for size type. We use 0 to indicate raw size (number of shares).
      - direction: integer code for direction (1=LONG/buy, 2=SHORT/sell).

    Important: We intentionally use raw share sizes (size_type=0) to avoid relying on enum values
    for notional-based ordering.

    Strategy logic (pair A/B):
      - Hedge ratio is rolling OLS slope b from compute_spread_indicators.
      - Compute base unit for A: units_A = notional_per_leg / price_A.
      - units_B = hedge_ratio * units_A (can be negative; sign determines buy/sell for B).

      Entry:
        - If zscore > entry_threshold: Short A (units_A), Long B (units_B).
        - If zscore < -entry_threshold: Long A (units_A), Short B (units_B).

      Exit:
        - If zscore crosses zero (sign change from previous bar): close positions.
        - If |zscore| > stop_threshold: stop-loss, close positions.

    Args:
        c: Order context (has attributes i, col, position_now, cash_now for the flexible wrapper).
        close_a, close_b: price arrays.
        zscore, hedge_ratio: indicator arrays (same length as price arrays).
        entry_threshold: entry z-score threshold (default 2.0).
        exit_threshold: (unused directly; kept for signature compatibility).
        stop_threshold: stop-loss threshold (default 3.0).
        notional_per_leg: fixed notional per leg in USD (default 10000.0).

    Returns:
        Tuple of (size, size_type, direction). Return (np.nan, 0, 0) to indicate no order.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 for asset A, 1 for asset B

    # Safely get current zscore and hedge_ratio
    try:
        cur_z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    try:
        cur_hr = float(hedge_ratio[i])
    except Exception:
        cur_hr = np.nan

    # If indicators are not available, do nothing
    if np.isnan(cur_z) or np.isnan(cur_hr):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current position for this asset (in base units). Use 0 if not provided
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Helper to return no order
    NO_ORDER = (np.nan, 0, 0)

    # Exit logic: if position exists and we meet exit conditions, close it
    if abs(pos_now) > 0:
        # Previous zscore (for cross-zero detection)
        prev_z = float(zscore[i - 1]) if i > 0 else np.nan

        crossed_zero = False
        if not np.isnan(prev_z):
            # Consider crossing when sign flips or when one of them is exactly zero and the other has opposite sign
            try:
                crossed_zero = (prev_z > 0 and cur_z <= 0) or (prev_z < 0 and cur_z >= 0)
            except Exception:
                crossed_zero = False

        stop_loss = abs(cur_z) > stop_threshold

        if crossed_zero or stop_loss:
            # Close current position fully
            size_to_close = abs(pos_now)
            # If current position is long (>0) we need to sell to close (SHORT direction=2)
            # If current position is short (<0) we need to buy to close (LONG direction=1)
            direction = 2 if pos_now > 0 else 1
            return (float(size_to_close), 0, int(direction))

        # Otherwise, keep position
        return NO_ORDER

    # No position currently: check entry conditions
    # Recompute base units for A (guard against zero price)
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return NO_ORDER

    units_a = notional_per_leg / price_a

    # Entry: short A, long B when zscore > entry_threshold
    if cur_z > entry_threshold:
        if col == 0:
            # Asset A: short
            return (float(units_a), 0, 2)
        else:
            # Asset B: long hedge_ratio * units_a (sign may flip)
            units_b = cur_hr * units_a
            if units_b == 0 or np.isnan(units_b):
                return NO_ORDER
            if units_b > 0:
                return (float(abs(units_b)), 0, 1)
            else:
                return (float(abs(units_b)), 0, 2)

    # Entry: long A, short B when zscore < -entry_threshold
    if cur_z < -entry_threshold:
        if col == 0:
            # Asset A: long
            return (float(units_a), 0, 1)
        else:
            # Asset B: short hedge_ratio * units_a
            units_b = cur_hr * units_a
            if units_b == 0 or np.isnan(units_b):
                return NO_ORDER
            if units_b > 0:
                # positive units_b means we need to short B for this leg
                return (float(abs(units_b)), 0, 2)
            else:
                # negative units_b means we need to long B
                return (float(abs(units_b)), 0, 1)

    # Otherwise, no order
    return NO_ORDER
