"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> Dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> Tuple[float,int,int]

Notes:
- Rolling OLS hedge ratio computed with expanding window up to hedge_lookback (uses only past data).
- Z-score computed using rolling mean/std over zscore_lookback (uses past data only).
- No numba usage.
"""
from typing import Any, Dict, Tuple

import numpy as np


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert input to a 1D numpy array.

    Accepts numpy arrays, pandas Series/DataFrame, lists. If input is 2D with
    a single column, it is flattened. If multi-column, the first column is taken.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Flatten single-column or take first column
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr[:, 0]
    # For higher dimensional input, ravel
    return arr.ravel()


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score of the spread.

    Args:
        close_a: Prices of asset A (1D array-like or single-column 2D).
        close_b: Prices of asset B (1D array-like or single-column 2D).
        hedge_lookback: Lookback for rolling OLS hedge ratio (use up to this many past points).
        zscore_lookback: Lookback for rolling mean/std to compute z-score.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-score values (same length as inputs)

    Implementation details:
        - At time t, hedge ratio is calculated using OLS on pairs from indices
          [max(0, t-hedge_lookback+1) .. t] (no future data used).
        - If the regressor has near-zero variance, the previous hedge ratio is carried forward
          (or 1.0 for the first element).
        - Z-score is computed as (spread - rolling_mean) / rolling_std where rolling
          stats use up to zscore_lookback past points (including current).
        - This function avoids producing NaNs by falling back to sensible defaults.
    """
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    if a.size != b.size:
        raise ValueError("close_a and close_b must have the same length")

    n = a.size

    # Preallocate
    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Initialize with a default hedge ratio of 1.0
    prev_hr = 1.0

    for t in range(n):
        # Rolling window for hedge ratio (use available past data up to hedge_lookback)
        start = max(0, t - hedge_lookback + 1)
        x = b[start : t + 1]  # regressor (price B)
        y = a[start : t + 1]  # dependent (price A)

        # Need at least 2 points to estimate slope; otherwise carry forward previous
        if x.size >= 2:
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom <= 1e-12:
                hr = prev_hr
            else:
                hr = ((x - x_mean) * (y - y_mean)).sum() / denom
                if not np.isfinite(hr):
                    hr = prev_hr
        else:
            hr = prev_hr

        hedge_ratio[t] = hr
        prev_hr = hr

        # Spread at time t (uses hedge_ratio estimated up to t)
        spread[t] = a[t] - hr * b[t]

        # Rolling mean/std for z-score (use available past data up to zscore_lookback)
        start_z = max(0, t - zscore_lookback + 1)
        window = spread[start_z : t + 1]
        if window.size >= 1:
            mu = window.mean()
            sigma = window.std(ddof=0)
            if sigma <= 1e-12:
                z = 0.0
            else:
                z = (spread[t] - mu) / sigma
        else:
            z = 0.0

        zscore[t] = z

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


def order_func(
    c: Any,
    close_a: Any,
    close_b: Any,
    zscore: Any,
    hedge_ratio: Any,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """Flexible order function for pairs trading.

    This function is called once per column (asset) per bar by the flexible wrapper.
    It returns a tuple (size, size_type, direction) where:
      - size: magnitude of the order (number of units)
      - size_type: integer code (we use 0 -> size in units)
      - direction: integer code for trade direction (0 -> long/buy, 1 -> short/sell)

    Logic:
      - Entry when zscore > entry_threshold: short A, long B
      - Entry when zscore < -entry_threshold: long A, short B
      - Exit when zscore crosses 0 (sign change) or |zscore| > stop_threshold
      - Position sizing: fixed notional_per_leg for asset A; asset B sized by hedge_ratio
        so that position_b = -hedge_ratio * position_a (opposite sign)

    The function uses only past and current bar data (c.i index) and c.position_now to
    decide whether to place an order.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    ca = _to_1d_array(close_a)
    cb = _to_1d_array(close_b)
    z = _to_1d_array(zscore)
    hr = _to_1d_array(hedge_ratio)

    # Bound-check index
    if i < 0 or i >= z.size:
        return (np.nan, 0, 0)

    z_t = float(z[i]) if np.isfinite(z[i]) else 0.0
    hr_t = float(hr[i]) if np.isfinite(hr[i]) else 1.0

    price_a = float(ca[i]) if ca.size > i else 1.0
    price_b = float(cb[i]) if cb.size > i else 1.0

    # Current position for this asset (units). May be 0.0
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Compute target units for asset A (anchor) based on fixed notional
    units_a = notional_per_leg / price_a if price_a != 0 else 0.0

    # Determine desired action based on z-score
    action_open = None
    if z_t > entry_threshold:
        action_open = "SHORT_A_LONG_B"
    elif z_t < -entry_threshold:
        action_open = "LONG_A_SHORT_B"

    # If currently flat, consider opening
    if abs(pos_now) < 1e-8:
        if action_open is None:
            # No entry signal for this asset
            return (np.nan, 0, 0)

        # Compute target positions (in units) for both assets
        if action_open == "SHORT_A_LONG_B":
            pos_a_target = -units_a
        else:
            pos_a_target = +units_a

        pos_b_target = -hr_t * pos_a_target

        if col == 0:
            size = abs(pos_a_target)
            direction = 0 if pos_a_target > 0 else 1
            return (float(size), 0, int(direction))
        else:
            size = abs(pos_b_target)
            direction = 0 if pos_b_target > 0 else 1
            return (float(size), 0, int(direction))

    # If we have a position, check for exit conditions
    if abs(z_t) > stop_threshold:
        if pos_now > 0:
            return (abs(pos_now), 0, 1)
        else:
            return (abs(pos_now), 0, 0)

    prev_z = float(z[i - 1]) if i > 0 and np.isfinite(z[i - 1]) else z_t
    if prev_z * z_t < 0:
        if pos_now > 0:
            return (abs(pos_now), 0, 1)
        else:
            return (abs(pos_now), 0, 0)

    if abs(z_t) <= exit_threshold:
        if pos_now > 0:
            return (abs(pos_now), 0, 1)
        else:
            return (abs(pos_now), 0, 0)

    return (np.nan, 0, 0)
