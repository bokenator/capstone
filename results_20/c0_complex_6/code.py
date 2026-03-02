"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- No numba usage
- Functions use numpy/pandas/scipy only

"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A as 1D numpy array.
        close_b: Prices for asset B as 1D numpy array.
        hedge_lookback: Window length for rolling OLS to estimate hedge ratio.
        zscore_lookback: Window length for rolling mean/std of the spread to compute z-score.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": np.ndarray of same length as inputs with rolling OLS slope.
            - "spread": np.ndarray spread = close_a - hedge_ratio * close_b
            - "zscore": np.ndarray z-score of the spread (rolling mean/std)

    Edge cases:
        - Windows containing NaNs are handled by dropping NaN pairs. If insufficient
          points exist in a window (less than 2), hedge_ratio is set to NaN.
        - Rolling std equal to zero produces NaN z-score.
    """
    # Validate inputs
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")
    n = len(a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (slope only) of A ~ B using scipy.stats.linregress
    # Use inclusive window ending at i (i.e., window = [i - lookback + 1, ..., i])
    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        window_a = a[start : i + 1]
        window_b = b[start : i + 1]

        # Drop pairs with NaNs
        mask = ~np.isnan(window_a) & ~np.isnan(window_b)
        if np.sum(mask) < 2:
            # Not enough data for regression
            hedge_ratio[i] = np.nan
            continue

        ya = window_a[mask]
        xb = window_b[mask]

        # If xb has zero variance, slope is undefined
        if np.allclose(xb, xb[0]):
            hedge_ratio[i] = np.nan
            continue

        lr = stats.linregress(xb, ya)
        slope = lr.slope
        hedge_ratio[i] = float(slope)

    # Compute spread using hedge ratio (elementwise)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = a[valid_hr] - hedge_ratio[valid_hr] * b[valid_hr]

    # Compute rolling mean and std of spread using pandas for convenience
    spread_series = pd.Series(spread)
    # Require full window to produce value (min_periods = zscore_lookback)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    zscore = (spread_series - roll_mean) / roll_std

    # Convert back to numpy arrays
    result = {
        "hedge_ratio": hedge_ratio,
        "spread": spread_series.values.astype(float),
        "zscore": zscore.values.astype(float),
    }
    return result


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
    """Order function for a pairs trading strategy (flexible multi-asset mode).

    The wrapper will call this function once per asset (col) per bar.

    Args:
        c: Order context. Exposes at least attributes:
            - i: current integer bar index
            - col: column index (0 for asset A, 1 for asset B)
            - position_now: current position size (in units) for this asset
            - cash_now or value_now: current cash/value (may not be used)
        close_a, close_b: price arrays for both assets
        zscore: array of spread z-scores
        hedge_ratio: array of rolling hedge ratios
        entry_threshold: threshold to enter trades (default 2.0)
        exit_threshold: threshold to exit when crossing (default 0.0)
        stop_threshold: absolute z-score at which to stop out (default 3.0)
        notional_per_leg: dollar notional used to size base leg

    Returns:
        Tuple (size, size_type, direction)
        - size: positive number (units) or np.nan for no order
        - size_type: integer enum for size interpretation. We use 1 to indicate absolute units.
        - direction: integer indicating buy (1) or sell (2).

    Notes:
        - We compute units for asset A as notional_per_leg / price_a. Asset B units follow
          the hedge ratio: units_B = hedge_ratio * units_A. This preserves the ratio of
          units between A and B while sizing the base leg by notional_per_leg.
        - We do not use vectorbt enums directly to keep this function pure Python.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive: ensure arrays are numpy and long enough
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    pos_now = float(getattr(c, "position_now", 0.0))

    # If indicators are NaN, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute unit sizes
    # Base units for asset A derived from fixed notional per leg
    if price_a <= 0:
        return (np.nan, 0, 0)

    units_a = float(notional_per_leg / price_a)
    units_b = float(hr * units_a)

    # Decide per-column behavior
    # size_type: 1 -> absolute units (we avoid using vbt enums directly here)
    SIZE_TYPE_UNITS = 1
    # VectorBt Direction enum mapping: use 1 for BUY/LONG and 2 for SELL/SHORT
    DIRECTION_BUY = 1
    DIRECTION_SELL = 2

    # Helper: return no order
    def no_order() -> Tuple[float, int, int]:
        return (np.nan, 0, 0)

    # Helper: close current position fully
    def close_position(size_in_units: float, current_pos: float) -> Tuple[float, int, int]:
        if current_pos == 0 or np.isnan(current_pos):
            return no_order()
        # If currently long (>0), place sell order to close; if short (<0), place buy order
        if current_pos > 0:
            return (abs(current_pos), SIZE_TYPE_UNITS, DIRECTION_SELL)
        else:
            return (abs(current_pos), SIZE_TYPE_UNITS, DIRECTION_BUY)

    # Exit conditions: stop-loss or crossing zero
    # Determine crossing zero using previous z-score
    crossed_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        # Crossing zero if signs differ (product < 0)
        if prev_z * z < 0:
            crossed_zero = True
        # Also if current exactly equals the exit_threshold
        if prev_z != z and z == exit_threshold:
            crossed_zero = True

    if abs(z) > stop_threshold:
        # Stop-loss: close if we have a position
        return close_position(units_a if col == 0 else units_b, pos_now)

    if crossed_zero:
        # Z-score crossed zero: close if we have a position
        return close_position(units_a if col == 0 else units_b, pos_now)

    # Entry conditions
    # Enter only if currently flat
    is_flat = np.isclose(pos_now, 0.0)

    if is_flat:
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                # Asset A: short units_a
                return (units_a, SIZE_TYPE_UNITS, DIRECTION_SELL)
            else:
                # Asset B: long units_b
                return (units_b, SIZE_TYPE_UNITS, DIRECTION_BUY)

        if z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                return (units_a, SIZE_TYPE_UNITS, DIRECTION_BUY)
            else:
                return (units_b, SIZE_TYPE_UNITS, DIRECTION_SELL)

    # No action by default
    return no_order()
