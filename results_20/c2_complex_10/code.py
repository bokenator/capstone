# Pairs trading strategy utilities for vectorbt
# Implements compute_spread_indicators and order_func

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
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    Args:
        close_a: 1-D array of close prices for asset A.
        close_b: 1-D array of close prices for asset B.
        hedge_lookback: Lookback window for rolling OLS regression to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": rolling hedge ratio (slope) as numpy array
            - "spread": spread series Price_A - hedge_ratio * Price_B
            - "zscore": z-score of the spread using rolling mean/std

    Notes:
        - Warm-up periods where not enough data is available are filled with np.nan.
        - Uses scipy.stats.linregress for OLS slope estimation on each rolling window.
    """
    # Validate inputs
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.size

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # If lookback is larger than available data, just return NaNs
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")

    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    # Rolling OLS to estimate hedge ratio (slope of regression Price_A ~ Price_B)
    # We compute slope for each window ending at index t (inclusive).
    if n >= hedge_lookback:
        for end in range(hedge_lookback - 1, n):
            start = end - hedge_lookback + 1
            y = close_a[start : end + 1]
            x = close_b[start : end + 1]

            # Skip windows with NaNs
            if np.isnan(x).any() or np.isnan(y).any():
                hedge_ratio[end] = np.nan
                continue

            # If x has zero variance, slope is undefined
            if np.allclose(x, x[0]):
                hedge_ratio[end] = np.nan
                continue

            # linregress returns slope, intercept, r_value, p_value, std_err
            try:
                slope, _, _, _, _ = linregress(x, y)
            except Exception:
                # Fallback to numpy polyfit (should rarely happen)
                try:
                    slope = np.polyfit(x, y, 1)[0]
                except Exception:
                    slope = np.nan

            hedge_ratio[end] = float(slope)

    # Compute spread using hedge_ratio aligned at the same index (elementwise)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std of spread
    spread_series = pd.Series(spread)
    roll_mean = (
        spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback)
        .mean()
        .to_numpy()
    )
    # Use population std (ddof=0) for z-score stability; if you prefer sample std use ddof=1
    roll_std = (
        spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback)
        .std(ddof=0)
        .to_numpy()
    )

    # Compute z-score, guard against zero std
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = ~np.isnan(spread) & ~np.isnan(roll_mean) & ~np.isnan(roll_std) & (roll_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - roll_mean[valid_mask]) / roll_std[valid_mask]

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
    """
    Order function for a two-asset pairs trading strategy (flexible multi-asset mode).

    This function is written to be used with a wrapper that calls it for each
    column (asset) on every bar. It returns a tuple (size, size_type, direction).

    Behavior:
      - Entry when zscore > entry_threshold: short A, long B
      - Entry when zscore < -entry_threshold: long A, short B
      - Exit when zscore crosses 0.0 (sign change) or when |zscore| > stop_threshold: close positions
      - Position sizing: fixed notional_per_leg per leg. For asset A units = notional / price_A.
        Asset B units = abs(hedge_ratio) * units_A (direction determined by signal).

    Notes:
      - Returns (np.nan, 0, 0) to indicate no order.
      - Uses raw integers for size_type and direction to avoid importing vectorbt enums.
        * size_type = 0 -> absolute size (units)
        * direction: use vectorbt.Direction semantics
            - 0 -> LongOnly (open long entries)
            - 1 -> ShortOnly (open short entries)
            - 2 -> Both (allow both directions; useful for exits)

    Args:
        c: Order context / flex context provided by vectorbt (must have attributes i, col, position_now)
        close_a: array of asset A close prices
        close_b: array of asset B close prices
        zscore: array with z-score values
        hedge_ratio: array with rolling hedge ratio values
        entry_threshold: threshold to enter trades (default 2.0)
        exit_threshold: threshold to exit trades (not used directly since we check crossing 0.0)
        stop_threshold: absolute z-score beyond which to immediately close positions
        notional_per_leg: fixed dollar notional per leg

    Returns:
        Tuple[size, size_type, direction]
    """
    # Direction constants (vectorbt.Direction semantics)
    DIR_LONG_ONLY = 0
    DIR_SHORT_ONLY = 1
    DIR_BOTH = 2

    # Defaults: no order
    NO_ORDER: Tuple[float, int, int] = (np.nan, 0, DIR_BOTH)

    # Extract minimal info from context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely read arrays (assume they are numpy arrays)
    price_a = float(close_a[i]) if (0 <= i < len(close_a)) else np.nan
    price_b = float(close_b[i]) if (0 <= i < len(close_b)) else np.nan

    z_now = float(zscore[i]) if (0 <= i < len(zscore)) else np.nan
    hr_now = float(hedge_ratio[i]) if (0 <= i < len(hedge_ratio)) else np.nan

    # Current position (units) for this column
    position_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Helper to produce an order that closes current position
    def close_position_order(pos_now: float) -> Tuple[float, int, int]:
        if pos_now == 0 or np.isnan(pos_now):
            return NO_ORDER
        # Always use DIR_BOTH for exits so that either side can be reduced
        return (abs(pos_now), 0, DIR_BOTH)

    # STOP-LOSS (highest priority): if |zscore| > stop_threshold -> close any open position
    if not np.isnan(z_now) and abs(z_now) > stop_threshold:
        # only produce close orders for columns that have open positions
        if abs(position_now) > 0:
            return close_position_order(position_now)
        return NO_ORDER

    # EXIT on z-score crossing 0.0 (sign change)
    prev_z = float(zscore[i - 1]) if (i - 1 >= 0 and i - 1 < len(zscore)) else np.nan
    crossed_zero = False
    if not np.isnan(prev_z) and not np.isnan(z_now):
        # Detect crossing through zero (including landing exactly at zero)
        if prev_z * z_now <= 0 and prev_z != 0:
            crossed_zero = True
    if crossed_zero:
        if abs(position_now) > 0:
            return close_position_order(position_now)
        return NO_ORDER

    # ENTRY conditions: only open if there is no existing position in this column
    pos_tolerance = 1e-12
    has_position = abs(position_now) > pos_tolerance

    # Avoid trading when price or hedge ratio is invalid
    if col == 0:
        price = price_a
    else:
        price = price_b

    if np.isnan(price) or price <= 0:
        return NO_ORDER

    # Compute base units for asset A (used to size both legs)
    if np.isnan(price_a) or price_a <= 0:
        # cannot size leg based on asset A
        return NO_ORDER

    units_a = notional_per_leg / price_a

    # Entry: z > entry_threshold -> SHORT A, LONG B
    if (not np.isnan(z_now)) and (z_now > entry_threshold):
        # For asset A (col 0): short
        if col == 0 and not has_position:
            size = float(units_a)
            return (size, 0, DIR_SHORT_ONLY)

        # For asset B (col 1): long, scaled by hedge ratio magnitude
        if col == 1 and not has_position:
            if np.isnan(hr_now) or hr_now == 0:
                return NO_ORDER
            size = float(abs(hr_now) * units_a)
            return (size, 0, DIR_LONG_ONLY)

        return NO_ORDER

    # Entry: z < -entry_threshold -> LONG A, SHORT B
    if (not np.isnan(z_now)) and (z_now < -entry_threshold):
        # For asset A (col 0): long
        if col == 0 and not has_position:
            size = float(units_a)
            return (size, 0, DIR_LONG_ONLY)

        # For asset B (col 1): short, scaled by hedge ratio magnitude
        if col == 1 and not has_position:
            if np.isnan(hr_now) or hr_now == 0:
                return NO_ORDER
            size = float(abs(hr_now) * units_a)
            return (size, 0, DIR_SHORT_ONLY)

        return NO_ORDER

    # No other action
    return NO_ORDER
