"""
Pairs trading strategy with rolling OLS hedge ratio and z-score entries/exits.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20)
- order_func(c, close_a, close_b, zscore, hedge_ratio,
             entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

Notes:
- No numba used.
- Uses scipy.stats.linregress for OLS hedge ratio per rolling window.
- Order function is stateless and works in flexible multi-asset mode used by vectorbt.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray, close_b: np.ndarray,
    hedge_lookback: int = 60, zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (array-like)
        close_b: Prices for asset B (array-like)
        hedge_lookback: Lookback window for rolling OLS (use up to available data before)
        zscore_lookback: Lookback window for rolling mean/std of spread

    Returns:
        Dict with keys:
            - "hedge_ratio": numpy array of hedge ratios (same length as inputs)
            - "zscore": numpy array of z-scores (same length as inputs)

    Implementation details:
        - Hedge ratio at time t is computed by OLS regression of A on B using
          data from max(0, t - hedge_lookback + 1) to t (inclusive). If fewer than
          2 valid observations are available, hedge ratio defaults to previous value or 0.0.
        - Spread = A - hedge_ratio * B
        - Rolling mean/std use pandas rolling with min_periods=1 to avoid NaNs early.
        - If rolling std is zero, z-score is set to 0.0 to avoid division by zero.
    """
    # Convert inputs to 1D numpy arrays
    close_a_arr = np.asarray(close_a, dtype=float).flatten()
    close_b_arr = np.asarray(close_b, dtype=float).flatten()

    if close_a_arr.shape[0] != close_b_arr.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = close_a_arr.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)

    # Rolling OLS for hedge ratio: use available data up to current index (no lookahead)
    for t in range(n):
        start = max(0, t - hedge_lookback + 1)
        y = close_a_arr[start : t + 1]
        x = close_b_arr[start : t + 1]

        # Remove any NaNs from the pair-wise data
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if x_valid.size >= 2:
            try:
                slope = float(linregress(x_valid, y_valid).slope)
            except Exception:
                slope = 0.0
        else:
            # Not enough data yet -> fallback to previous hedge ratio if available, else 0
            slope = hedge_ratio[t - 1] if t > 0 else 0.0

        hedge_ratio[t] = slope

    # Compute spread
    spread = close_a_arr - hedge_ratio * close_b_arr

    # Rolling mean/std for z-score (use population std to avoid NaNs on small windows)
    spread_sr = pd.Series(spread)
    rolling_mean = spread_sr.rolling(window=zscore_lookback, min_periods=1).mean()
    rolling_std = spread_sr.rolling(window=zscore_lookback, min_periods=1).std(ddof=0)

    # Avoid division by zero
    rolling_std_values = rolling_std.to_numpy()
    rolling_std_values[rolling_std_values == 0] = np.nan

    zscore = (spread - rolling_mean.to_numpy()) / rolling_std_values

    # Where std was zero, set zscore to 0.0 (no deviation)
    zscore = np.where(np.isnan(zscore), 0.0, zscore)

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
    """
    Order function for a pairs trading strategy (flexible multi-asset mode).

    This function is called once per asset (column) per bar. It must return a tuple
    (size, size_type, direction) where:
        - size: absolute order size (float). Return np.nan to indicate no order.
        - size_type: integer code for size type (use 0 for absolute units).
        - direction: integer direction (1 for buy/long, 2 for sell/short, 0 for no order).

    Strategy rules implemented:
        - Entry when |zscore| > entry_threshold:
            - zscore > entry_threshold: Short A, Long B
            - zscore < -entry_threshold: Long A, Short B
        - Exit when zscore crosses zero (sign change) OR when |zscore| > stop_threshold (stop-loss)
        - Position sizing:
            - Asset A units = notional_per_leg / price_A
            - Asset B units = hedge_ratio * Asset_A_units

    The function is written to be robust with different context objects. It uses c.i and c.col
    to determine the current bar and column. It reads c.position_now when available to determine
    whether the current asset already has a position.
    """
    # Helper: safe attribute getter
    def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name) if hasattr(obj, name) else default

    i = int(_safe_getattr(c, "i", 0))
    col = int(_safe_getattr(c, "col", 0))

    # Determine current position for this column (units, signed)
    pos_now = _safe_getattr(c, "position_now", None)
    if pos_now is None:
        # Try last_position array if available
        last_pos = _safe_getattr(c, "last_position", None)
        if last_pos is not None:
            try:
                pos_now = float(last_pos[col])
            except Exception:
                pos_now = 0.0
        else:
            # Fallback
            pos_now = 0.0

    # Convert arrays to numpy for indexing
    close_a_arr = np.asarray(close_a, dtype=float).flatten()
    close_b_arr = np.asarray(close_b, dtype=float).flatten()
    z_arr = np.asarray(zscore, dtype=float).flatten()
    hr_arr = np.asarray(hedge_ratio, dtype=float).flatten()

    # Basic safety checks
    if i < 0 or i >= z_arr.shape[0]:
        return (np.nan, 0, 0)

    price_a = float(close_a_arr[i])
    price_b = float(close_b_arr[i])
    z = float(z_arr[i])
    hr = float(hr_arr[i])

    # If indicators invalid, do nothing
    if np.isnan(z) or np.isnan(hr) or price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    # Helper to close current position for this asset
    def _close_position(size_now: float) -> Tuple[float, int, int]:
        # size_now is signed units (positive = long, negative = short)
        size = float(abs(size_now))
        if size == 0 or np.isnan(size):
            return (np.nan, 0, 0)
        # To close a long position (size_now>0) we sell -> direction = 2
        # To close a short position (size_now<0) we buy -> direction = 1
        direction = 1 if size_now < 0 else 2
        return (size, 0, int(direction))

    # STOP-LOSS: if |zscore| > stop_threshold, close any existing position
    if abs(z) > stop_threshold:
        if pos_now is not None and pos_now != 0:
            return _close_position(float(pos_now))
        else:
            return (np.nan, 0, 0)

    # EXIT on mean reversion: z-score crosses zero (sign change)
    z_prev = float(z_arr[i - 1]) if i > 0 else np.nan
    crossed_zero = False
    if not np.isnan(z_prev):
        if (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold):
            crossed_zero = True

    if crossed_zero:
        if pos_now is not None and pos_now != 0:
            return _close_position(float(pos_now))
        else:
            return (np.nan, 0, 0)

    # ENTRY: when |zscore| > entry_threshold and we don't currently hold position for this asset
    if abs(z) > entry_threshold:
        # Only open if we currently have no position
        if pos_now is None or pos_now == 0:
            # Compute base sizes
            # Asset A units (base leg)
            size_a = float(notional_per_leg / price_a)
            # Asset B scaled by hedge ratio (can be negative in degenerate cases), use absolute multiplier
            size_b = float(abs(hr) * size_a)

            # z > threshold: SHORT A, LONG B
            if z > entry_threshold:
                if col == 0:
                    # Short A -> sell
                    return (size_a, 0, 2)
                else:
                    # Long B -> buy
                    return (size_b, 0, 1)

            # z < -threshold: LONG A, SHORT B
            if z < -entry_threshold:
                if col == 0:
                    # Long A -> buy
                    return (size_a, 0, 1)
                else:
                    # Short B -> sell
                    return (size_b, 0, 2)

    # Otherwise no order
    return (np.nan, 0, 0)
