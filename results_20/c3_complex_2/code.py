"""
Pairs trading indicator and order functions for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20)
- order_func(c, close_a, close_b, zscore, hedge_ratio,
             entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

Notes:
- Rolling OLS hedge ratio computed using past window up to current bar (no lookahead).
- Rolling mean/std for z-score use available history (expanding at the beginning).
- Orders returned in Value size_type (1) and Direction: 1=BUY (long), 2=SELL (short).
- When entering a trade: asset A is sized by notional_per_leg; asset B units are scaled by hedge_ratio to achieve units_B ~= hedge_ratio * units_A (opposite sign).
- Exit on z-score crossing zero or stop-loss (|z| > stop_threshold).

This file contains only pure Python (no numba usage).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert various array-like inputs to a 1D numpy float array.

    If x is 2D, we try to extract a single column (first column). This makes the
    function robust to receiving pandas DataFrames or 2D numpy arrays.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.array([float(arr)])
    if arr.ndim == 1:
        return arr.astype(float)
    # If 2D or higher, try to pick the first column
    if arr.ndim >= 2:
        # If it's shape (n, 1) -> squeeze
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        # If it's (n, m) with m >= 2, assume first column corresponds to the requested series
        return arr[:, 0].astype(float)
    # fallback
    return arr.ravel().astype(float)


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio and z-score for a pairs trading strategy.

    Args:
        close_a: Close prices for asset A (array-like).
        close_b: Close prices for asset B (array-like).
        hedge_lookback: Lookback window (in bars) for rolling OLS hedge ratio.
                        For early bars where less data is available, an expanding
                        regression is used.
        zscore_lookback: Lookback for rolling mean/std of the spread. Uses an
                         expanding window at the beginning to avoid NaNs.

    Returns:
        A dict with keys:
        - "hedge_ratio": np.ndarray of hedge ratios (slope of regressing A on B)
        - "zscore": np.ndarray of z-score values for the spread
        - "spread": np.ndarray of spread values (A - hedge_ratio * B)

    Notes:
        All outputs have the same length as the input price arrays. Computations
        are causal (use only data up to and including the current index).
    """
    # Convert inputs to 1D numpy arrays
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling / expanding OLS (regress A on B) -> slope is hedge ratio
    for i in range(n):
        start = 0 if i + 1 <= hedge_lookback else i + 1 - hedge_lookback
        xa = b[start : i + 1]
        ya = a[start : i + 1]

        # Need at least 2 points to estimate slope
        if xa.size < 2 or np.all(np.isnan(xa)) or np.all(np.isnan(ya)):
            # Fallback to previous hedge ratio if available, else 0.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 0.0
            continue

        # If B is constant in the window, slope is undefined; fallback to previous or 0
        if np.nanstd(xa) == 0.0:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 0.0
            continue

        # Use scipy linregress for a robust slope estimate (no future data used)
        try:
            lr = linregress(xa, ya)
            slope = lr.slope if np.isfinite(lr.slope) else (hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 0.0)
        except Exception:
            slope = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 0.0

        hedge_ratio[i] = float(slope)

    # Compute spread using current hedge_ratio (causal)
    spread = a - hedge_ratio * b

    # Rolling (expanding at start) mean and std for z-score
    mean = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = 0 if i + 1 <= zscore_lookback else i + 1 - zscore_lookback
        w = spread[start : i + 1]
        # Use nan-aware stats in case of NaNs
        mean[i] = float(np.nanmean(w)) if w.size > 0 else 0.0
        # Population std (ddof=0) to match many trading implementations
        s = float(np.nanstd(w, ddof=0)) if w.size > 0 else 0.0
        std[i] = s

    # Compute z-score, handle zero-std gracefully
    zscore = np.zeros(n, dtype=float)
    nonzero = std > 0
    # Safely compute only for 1D arrays
    zscore[nonzero] = (spread[nonzero] - mean[nonzero]) / std[nonzero]
    zscore[~nonzero] = 0.0

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset backtest.

    This function is designed to be called once per asset per bar. The context
    `c` is expected to provide at least `.i` (current integer index) and
    `.col` (column index 0 for asset A, 1 for asset B). In the flexible wrapper
    used by the backtest runner, the context will also include
    `.position_now` (current units held for that asset) and `.cash_now`.

    Returns a tuple (size, size_type, direction):
        - size: positive float size (in the units specified by size_type). If
          size is NaN, no order will be placed for this asset.
        - size_type: integer code. We use 1 to represent Value-based sizing
          (i.e., size is in quote currency / notional).
        - direction: integer code. We use 1 for BUY (long) and 2 for SELL (short).

    Position sizing and signals:
        - Entry (zscore > entry_threshold): SHORT A, LONG B
        - Entry (zscore < -entry_threshold): LONG A, SHORT B
        - Exit: zscore crosses 0.0 (sign change) or |zscore| > stop_threshold
        - Asset A target units are sized by notional_per_leg / price_a
        - Asset B units are set to hedge_ratio * units_A (opposite sign)

    The function returns the notional delta required to move from the current
    position to the desired target (size_type=1, value). Direction expresses
    buy/sell for that notional.
    """
    # Helper to safely extract attributes from context
    def _get_attr(obj: Any, names, default=None):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    # Extract index and column
    i = int(_get_attr(c, ["i", "index"], 0))
    col = int(_get_attr(c, ["col", "column"], 0))

    # Safely get current position in units (may be named differently in contexts)
    pos_now = _get_attr(c, ["position_now", "position", "last_position"], 0.0)
    # If last_position is an array-like, pick the appropriate column
    if isinstance(pos_now, (list, tuple, np.ndarray)):
        try:
            pos_now = float(pos_now[col])
        except Exception:
            pos_now = 0.0
    else:
        try:
            pos_now = float(pos_now)
        except Exception:
            pos_now = 0.0

    # Read prices and indicators at current index (use causal values only)
    price_a = float(_to_1d_array(close_a)[i])
    price_b = float(_to_1d_array(close_b)[i])

    z = float(_to_1d_array(zscore)[i])

    # Hedge ratio might be NaN if not computable; fallback to 0
    hr_arr = _to_1d_array(hedge_ratio)
    h = float(hr_arr[i]) if (i >= 0 and not np.isnan(hr_arr[i])) else 0.0

    # Units for asset A based on notional per leg
    units_a_target_mag = (notional_per_leg / price_a) if price_a > 0 else 0.0

    # Determine previous z for crossover detection (use same causal rule)
    z_prev = float(_to_1d_array(zscore)[i - 1]) if i > 0 else z

    # Simple small tolerance
    tiny = 1e-12

    # Determine desired units for this column
    desired_units = None

    is_in_position = abs(pos_now) > tiny

    # If we're currently in a position for this asset, check for exit conditions
    if is_in_position:
        # Stop-loss: close both legs when |z| > stop_threshold
        if abs(z) > stop_threshold:
            desired_units = 0.0
        # Exit on crossing zero: previous sign different from current
        elif z_prev * z < 0 and abs(z - z_prev) > 0:
            desired_units = 0.0
        else:
            # Stay in current position for this asset
            return (np.nan, 0, 0)
    else:
        # Not in position -> check entry signals
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                desired_units = -units_a_target_mag
            else:
                desired_units = +abs(h) * units_a_target_mag
        elif z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                desired_units = +units_a_target_mag
            else:
                desired_units = -abs(h) * units_a_target_mag
        else:
            # No entry signal
            return (np.nan, 0, 0)

    # Compute units delta we need to trade
    delta_units = float(desired_units) - float(pos_now)

    # If no meaningful change, return no order
    price = price_a if col == 0 else price_b
    notional_change = delta_units * price

    if abs(notional_change) <= tiny:
        return (np.nan, 0, 0)

    # Use Value-based sizing: size is absolute notional to trade
    size = float(abs(notional_change))
    size_type = 1  # Value sizing (notional)
    # Direction: 1 = BUY (long), 2 = SELL (short)
    direction = 1 if notional_change > 0 else 2

    return (size, int(size_type), int(direction))
