"""
Pairs trading utility functions for vectorbt backtesting.

Provides:
- compute_spread_indicators: compute rolling OLS hedge ratio and spread z-score
- order_func: flexible order function (no numba) compatible with the provided
  backtest runner's flexible multi-asset wrapper.

Do NOT use numba.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Import enums for size_type and direction so we can return proper integer codes
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and spread z-score for a pair of assets.

    Args:
        close_a: Close prices for asset A (1D array-like)
        close_b: Close prices for asset B (1D array-like)
        hedge_lookback: Lookback window (in bars) for rolling OLS (slope)
        zscore_lookback: Lookback window (in bars) for rolling mean/std of spread

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of rolling hedge ratios (same length as inputs)
            - 'spread': np.ndarray of spread values (price_a - hedge_ratio * price_b)
            - 'rolling_mean': rolling mean of spread
            - 'rolling_std': rolling std of spread
            - 'zscore': rolling z-score of spread
    """
    # Convert to numpy arrays of float
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    if n == 0:
        # Return empty arrays
        return {
            "hedge_ratio": np.array([], dtype=float),
            "spread": np.array([], dtype=float),
            "rolling_mean": np.array([], dtype=float),
            "rolling_std": np.array([], dtype=float),
            "zscore": np.array([], dtype=float),
        }

    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be a positive integer")
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be a positive integer")

    w = int(hedge_lookback)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # If not enough data for even a single window, return hedgeratio full NaN
    if n >= w:
        # Valid windows are those where both a and b are not NaN for all w samples.
        valid_pair = (~np.isnan(a)) & (~np.isnan(b))

        # Replace NaNs with 0 for cumulative sums; we'll mask windows that are incomplete
        a_filled = np.where(np.isnan(a), 0.0, a)
        b_filled = np.where(np.isnan(b), 0.0, b)

        csum_a = np.concatenate(([0.0], np.cumsum(a_filled)))
        csum_b = np.concatenate(([0.0], np.cumsum(b_filled)))
        csum_ab = np.concatenate(([0.0], np.cumsum(a_filled * b_filled)))
        csum_b2 = np.concatenate(([0.0], np.cumsum(b_filled * b_filled)))

        csum_valid = np.concatenate(([0], np.cumsum(valid_pair.astype(int))))

        # Windowed sums (length = n - w + 1)
        sum_a = csum_a[w:] - csum_a[:-w]
        sum_b = csum_b[w:] - csum_b[:-w]
        sum_ab = csum_ab[w:] - csum_ab[:-w]
        sum_b2 = csum_b2[w:] - csum_b2[:-w]
        count_valid = csum_valid[w:] - csum_valid[:-w]

        # Compute slope (hedge ratio) where window is fully valid
        mask_full = count_valid == w
        if mask_full.any():
            mean_a = sum_a[mask_full] / w
            mean_b = sum_b[mask_full] / w

            s_ab = sum_ab[mask_full] - w * mean_a * mean_b
            s_b2 = sum_b2[mask_full] - w * mean_b * mean_b

            # Avoid division by zero
            valid_slope_mask = s_b2 != 0.0
            slopes = np.full(mask_full.sum(), np.nan, dtype=float)
            slopes[valid_slope_mask] = s_ab[valid_slope_mask] / s_b2[valid_slope_mask]

            # Place slopes into hedge_ratio aligned with right edge of window
            hedge_ratio[w - 1 :][mask_full] = slopes

    # Compute spread: a - hedge_ratio * b
    spread = a - hedge_ratio * b

    # Compute rolling mean/std of spread using pandas (to get min_periods behavior easily)
    spread_ser = pd.Series(spread)
    roll_mean = spread_ser.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).mean()
    # Use population std (ddof=0)
    roll_std = spread_ser.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).std(ddof=0)

    # z-score
    zscore_ser = (spread_ser - roll_mean) / roll_std

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": roll_mean.values.astype(float),
        "rolling_std": roll_std.values.astype(float),
        "zscore": zscore_ser.values.astype(float),
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
    Order function for flexible multi-asset backtest (no numba).

    The runner wraps this function to call it per-column using a simulated context
    that provides `i` (bar index), `col` (column index: 0 for asset A, 1 for asset B),
    and `position_now` (current position in units).

    Returns a tuple (size, size_type, direction) where:
      - size: number of units to trade (positive float). If np.nan, no order is placed.
      - size_type: integer code for size type (use SizeType.Amount -> units)
      - direction: integer code for order direction (Direction.LongOnly=0, ShortOnly=1, Both=2)
    """
    # Safety conversions
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    price = close_a[i] if col == 0 else close_b[i]

    # If price or indicators are NaN, do nothing
    if np.isnan(price) or np.isnan(zscore[i]) or np.isnan(hedge_ratio[i]):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    h = float(hedge_ratio[i])

    # Current position (units). Context may provide position_now; fall back to 0.
    pos_now = float(getattr(c, "position_now", 0.0))

    # Determine previous zscore for crossing detection
    prev_z = np.nan
    if i > 0:
        prev_z = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else np.nan

    # Crossing detection: previous and current have opposite signs (treat zero carefully)
    crossed_zero = False
    if not np.isnan(prev_z) and not np.isnan(z):
        if np.sign(prev_z) != np.sign(z) and prev_z != z:
            crossed_zero = True

    # Stop-loss detection
    stop_loss = abs(z) > float(stop_threshold)

    # If stop-loss or crossing zero and we have a position, close it using target amount = 0
    if (stop_loss or crossed_zero) and (not np.isclose(pos_now, 0.0)):
        # Use TargetAmount with target 0 and allow both directions to ensure position is closed
        return (0.0, int(SizeType.TargetAmount), int(Direction.Both))

    # Entry logic (only enter if flat)
    flat = np.isclose(pos_now, 0.0)

    # Base units (for asset A) used to compute B units via hedge ratio
    price_a = float(close_a[i])
    base_units = 0.0
    if price_a > 0.0:
        base_units = float(notional_per_leg) / price_a

    # Long/short decisions
    if z > float(entry_threshold):
        # SHORT A, LONG B
        if flat:
            if col == 0:
                # Short A
                size = base_units
                return (float(size), int(SizeType.Amount), int(Direction.ShortOnly))
            else:
                # Long B: units = hedge_ratio * base_units
                size = abs(h) * base_units
                return (float(size), int(SizeType.Amount), int(Direction.LongOnly))

    if z < -float(entry_threshold):
        # LONG A, SHORT B
        if flat:
            if col == 0:
                # Long A
                size = base_units
                return (float(size), int(SizeType.Amount), int(Direction.LongOnly))
            else:
                # Short B
                size = abs(h) * base_units
                return (float(size), int(SizeType.Amount), int(Direction.ShortOnly))

    # No order
    return (np.nan, 0, 0)
