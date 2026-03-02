"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold=2.0, exit_threshold=0.0, stop_threshold=3.0, notional_per_leg=10000.0) -> tuple

Notes:
- Rolling OLS hedge ratio computed using past data up to current bar. For early bars (fewer than hedge_lookback points) a regression is still performed on available points (min 2). This avoids NaNs early.
- Z-score uses rolling mean/std on past zscore_lookback samples (including current). Std==0 handled safely.
- order_func returns (size, size_type, direction) where size is number of units to trade (positive), size_type and direction are integers expected by vectorbt's order_nb.
  - When no order is to be placed, returns (np.nan, 0, 0).
- SizeType and Direction integer codes: we use size_type=0 indicating absolute share size, direction: 1=buy, 2=sell.

CRITICAL: No numba usage anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def _to_1d_array(obj) -> np.ndarray:
    """Convert various input types (Series, DataFrame, numpy arrays) to a 1D numpy array.

    If a DataFrame with multiple columns is passed, the caller is expected to handle it.
    """
    # pandas Series
    if isinstance(obj, pd.Series):
        return obj.values.astype(float)
    # pandas DataFrame -> if single column, squeeze, otherwise return as 2D array
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].values.astype(float)
        return obj.values.astype(float)
    # numpy array
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr.reshape(1).astype(float)
    return arr.astype(float)


def compute_spread_indicators(
    close_a: Any,
    close_b: Any = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of asset prices.

    This function is flexible in input types: close_a/close_b can be numpy arrays, pandas Series,
    or if close_a is a DataFrame containing two columns, close_b can be omitted.

    Args:
        close_a: 1D array-like for asset A closes, or DataFrame with 2 columns (A and B)
        close_b: 1D array-like for asset B closes (optional if close_a is DataFrame)
        hedge_lookback: lookback window for rolling OLS (use up to available history for early bars)
        zscore_lookback: lookback window for rolling mean/std of the spread

    Returns:
        Dict with keys: 'hedge_ratio', 'spread', 'zscore'. Each value is a 1D numpy array of length len(close_a).
    """
    # Allow passing a single DataFrame with two columns
    if close_b is None and isinstance(close_a, pd.DataFrame) and close_a.shape[1] >= 2:
        # Take first two columns
        close_a_ser = close_a.iloc[:, 0]
        close_b_ser = close_a.iloc[:, 1]
        close_a_arr = _to_1d_array(close_a_ser)
        close_b_arr = _to_1d_array(close_b_ser)
    else:
        # If close_a is DataFrame with >1 columns and close_b provided, assume close_a is one column of frame
        if isinstance(close_a, pd.DataFrame):
            # Try to squeeze to 1d if possible
            if close_a.shape[1] == 1:
                close_a_arr = _to_1d_array(close_a)
            else:
                # Use first column if close_b provided separately
                close_a_arr = _to_1d_array(close_a.iloc[:, 0])
        else:
            close_a_arr = _to_1d_array(close_a)

        if isinstance(close_b, pd.DataFrame):
            if close_b.shape[1] == 1:
                close_b_arr = _to_1d_array(close_b)
            else:
                close_b_arr = _to_1d_array(close_b.iloc[:, 0])
        else:
            close_b_arr = _to_1d_array(close_b)

    # If arrays are 2D but with one column, squeeze
    if close_a_arr.ndim == 2 and close_a_arr.shape[1] == 1:
        close_a_arr = close_a_arr.ravel()
    if close_b_arr.ndim == 2 and close_b_arr.shape[1] == 1:
        close_b_arr = close_b_arr.ravel()

    if close_a_arr.ndim != 1 or close_b_arr.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays or Series-like")

    # Align lengths: if inputs are Series with indexes, the caller should have passed aligned arrays.
    if len(close_a_arr) != len(close_b_arr):
        # Try to align if pandas Series with indexes were passed originally
        try:
            ca = pd.Series(close_a).astype(float)
            cb = pd.Series(close_b).astype(float)
            ca, cb = ca.align(cb, join="inner")
            close_a_arr = ca.values
            close_b_arr = cb.values
        except Exception:
            raise ValueError("close_a and close_b must have the same length after alignment")

    n = len(close_a_arr)
    if n == 0:
        return {"hedge_ratio": np.array([]), "spread": np.array([]), "zscore": np.array([])}

    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Rolling OLS: regress price_a on price_b using window ending at i (inclusive)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b_arr[start : i + 1]
        y = close_a_arr[start : i + 1]

        # Need at least 2 finite points to fit slope
        if x.size >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            x_mean = x.mean()
            y_mean = y.mean()
            x_center = x - x_mean
            y_center = y - y_mean
            denom = (x_center * x_center).sum()
            if denom > 0:
                slope = (x_center * y_center).sum() / denom
            else:
                slope = 0.0
        else:
            slope = 0.0

        hedge_ratio[i] = float(slope)

    # Compute spread (causal)
    spread = close_a_arr - hedge_ratio * close_b_arr

    # Rolling mean and std for spread
    for i in range(n):
        start = max(0, i - zscore_lookback + 1)
        window = spread[start : i + 1]
        if window.size > 0 and np.isfinite(window).all():
            mu = window.mean()
            sigma = window.std(ddof=0)
            if sigma == 0 or not np.isfinite(sigma):
                z = 0.0
            else:
                z = (spread[i] - mu) / sigma
        else:
            z = 0.0
        zscore[i] = float(z)

    return {"hedge_ratio": hedge_ratio, "spread": spread, "zscore": zscore}


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
    Order function for vectorbt flexible multi-asset mode.

    See earlier docstring in first submission for detailed behavior.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(np.asarray(close_a)[i])
    price_b = float(np.asarray(close_b)[i])

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else 0.0

    pos_now = float(getattr(c, "position_now", 0.0))

    # Default targets
    target_a = 0.0
    target_b = 0.0

    # Stop-loss: close both legs immediately
    if np.isfinite(z) and abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        if z > entry_threshold:
            target_a = -notional_per_leg / price_a if price_a > 0 else 0.0
            target_b = -hr * target_a
        elif z < -entry_threshold:
            target_a = notional_per_leg / price_a if price_a > 0 else 0.0
            target_b = -hr * target_a
        else:
            prev_z = float(zscore[i - 1]) if i > 0 else 0.0
            crossed_zero = (prev_z * z) < 0 if (i > 0 and np.isfinite(prev_z) and np.isfinite(z)) else False
            if crossed_zero:
                target_a = 0.0
                target_b = 0.0
            else:
                # No action
                return (np.nan, 0, 0)

    # Select target based on column
    target = float(target_a) if col == 0 else float(target_b)

    delta = target - pos_now

    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    size_to_trade = abs(delta)
    size_type = 0  # absolute units
    direction = 1 if delta > 0 else 2  # 1=buy, 2=sell

    return (float(size_to_trade), int(size_type), int(direction))
