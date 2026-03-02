# Pairs trading strategy utilities for vectorbt

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread, and z-score for a pair of assets.

    Args:
        close_a: 1D array of close prices for asset A
        close_b: 1D array of close prices for asset B
        hedge_lookback: Lookback window (in bars) for the rolling OLS hedge ratio
        zscore_lookback: Lookback window for rolling mean/std used to compute z-score

    Returns:
        Dictionary with keys:
            - 'hedge_ratio': ndarray of rolling hedge ratios (slope of A ~ B)
            - 'zscore': ndarray of z-score values for the spread

    Notes:
        - Where there isn't enough data for a calculation, values are NaN.
        - Uses scipy.stats.linregress for the OLS slope.
    """

    # Validate inputs
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS slope (hedge ratio) of A ~ B
    # For each window we regress A = slope * B + intercept and take slope
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be a positive integer")
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be a positive integer")

    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        y = close_a[start_idx : end_idx + 1]
        x = close_b[start_idx : end_idx + 1]

        # If there are NaNs in the window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[end_idx] = np.nan
            continue

        # If x is constant, slope is undefined -> NaN
        if np.allclose(x, x[0]):
            hedge_ratio[end_idx] = np.nan
            continue

        try:
            res = stats.linregress(x, y)
            hedge_ratio[end_idx] = float(res.slope)
        except Exception:
            hedge_ratio[end_idx] = np.nan

    # Compute spread: spread = A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std of spread for z-score
    spread_series = pd.Series(spread)
    # Require full lookback for reliable z-score (min_periods = zscore_lookback)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to avoid dividing by zero when sample std with small windows
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = (
        ~np.isnan(spread_series.values)
        & ~np.isnan(roll_mean.values)
        & (roll_std.values > 0)
    )
    zscore[valid_mask] = (
        (spread_series.values[valid_mask] - roll_mean.values[valid_mask])
        / roll_std.values[valid_mask]
    )

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset vectorbt backtest (pairs trading).

    This function is called once per asset (column) per bar. The provided context
    `c` is expected to have attributes:
      - c.i : current bar index
      - c.col : which column (0 for asset A, 1 for asset B)
      - c.position_now : current position (units) for this asset
      - c.cash_now or c.value_now : available cash (not used here)

    The function returns a tuple (size, size_type, direction) where:
      - size: numeric size for the order (meaning depends on size_type)
      - size_type: integer code for SizeType (1 = Value)
      - direction: integer code for Direction (0 = LongOnly, 1 = ShortOnly, 2 = Both)

    Sizing and behavior implemented:
      - Entry: when zscore > entry_threshold -> short A, long B
               when zscore < -entry_threshold -> long A, short B
        Orders use fixed notional per leg (size_type = Value, size = notional_per_leg)

      - Exit: when zscore crosses zero (sign change) -> close positions
              when abs(zscore) > stop_threshold -> stop-loss, close positions
        Close orders are implemented using SizeType.TargetValue with target 0

    Note: We avoid using numba or vectorbt-specific enums here and return simple
    integer codes expected by vectorbt's order constructor.
    """

    # Numeric codes from vectorbt docs (avoiding direct enum imports):
    SIZE_TYPE_VALUE = 1  # SizeType.Value
    SIZE_TYPE_TARGET_VALUE = 4  # SizeType.TargetValue
    DIRECTION_LONG = 0  # LongOnly
    DIRECTION_SHORT = 1  # ShortOnly
    DIRECTION_BOTH = 2  # Both

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, DIRECTION_BOTH)

    z_i = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Check stop-loss first
    if np.isfinite(z_i) and abs(z_i) > stop_threshold:
        if pos_now != 0.0:
            # Close position to target value 0
            return (0.0, SIZE_TYPE_TARGET_VALUE, DIRECTION_BOTH)
        else:
            return (np.nan, 0, DIRECTION_BOTH)

    # Check zero crossing to exit (requires previous value)
    if pos_now != 0.0 and np.isfinite(prev_z) and np.isfinite(z_i) and (prev_z * z_i < 0):
        return (0.0, SIZE_TYPE_TARGET_VALUE, DIRECTION_BOTH)

    # ENTRY CONDITIONS
    # Only enter when currently flat on this asset
    if pos_now == 0.0 and np.isfinite(z_i):
        # Short A, Long B when zscore > entry_threshold
        if z_i > entry_threshold:
            if col == 0:
                # Asset A: short
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)
            else:
                # Asset B: long
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)

        # Long A, Short B when zscore < -entry_threshold
        if z_i < -entry_threshold:
            if col == 0:
                # Asset A: long
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)
            else:
                # Asset B: short
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)

    # No order by default
    return (np.nan, 0, DIRECTION_BOTH)
