# pairs_strategy.py
"""
Pairs trading strategy utilities
Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (windowed, expands for warmup)
- Z-score computed with rolling mean/std
- Order function for flexible vectorbt.from_order_func usage

This module is pure Python / NumPy and does not use numba.
"""
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert various input types to 1D numpy array of floats.

    Handles:
      - numpy arrays (1D)
      - lists
      - pandas Series
      - pandas DataFrame (returns first numeric column)
    """
    if isinstance(x, np.ndarray):
        arr = x.astype(float)
        if arr.ndim == 1:
            return arr
        # If 2D with shape (n,1) return flattened
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        # If 2D with shape (n,2) leave as-is for caller to handle
        return arr.ravel()

    if isinstance(x, pd.Series):
        return x.values.astype(float)

    if isinstance(x, pd.DataFrame):
        # Try to find a column that represents 'close'
        # 1) MultiIndex columns where second level is 'close'
        if isinstance(x.columns, pd.MultiIndex):
            for col in x.columns:
                if str(col[-1]).lower() == "close":
                    return x[col].values.astype(float)
            # Try first column
            return x.iloc[:, 0].values.astype(float)

        # 2) Single-level columns
        # Prefer a column named 'close' (case-insensitive)
        for col in x.columns:
            if str(col).lower() == "close":
                return x[col].values.astype(float)

        # Otherwise select first numeric column
        numeric_cols = x.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            return x[numeric_cols[0]].values.astype(float)

        # Fallback: take first column and cast
        return x.iloc[:, 0].values.astype(float)

    # Fallback for lists/tuples/etc.
    return np.asarray(x, dtype=float)


def compute_spread_indicators(
    close_a: Any,
    close_b: Any = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    The function is flexible in input types:
      - close_a and close_b can be numpy arrays, lists, pandas Series
      - The first argument may also be a pandas DataFrame with two columns (A, B)
        in which case the second argument can be omitted.

    Args:
        close_a: Prices of asset A or a DataFrame containing both series
        close_b: Prices of asset B (optional if first arg is a 2-col DataFrame)
        hedge_lookback: Lookback for rolling OLS regression (number of bars)
        zscore_lookback: Lookback for rolling mean/std used to compute z-score

    Returns:
        Dictionary with keys:
            - "zscore": np.ndarray of z-scores (same length as inputs)
            - "hedge_ratio": np.ndarray of hedge ratios (same length as inputs)
            - "spread": np.ndarray of spread values
            - "rolling_mean": np.ndarray of rolling means
            - "rolling_std": np.ndarray of rolling stds

    Causality (no lookahead): All computations use only past and present data points
    up to index t when producing values at t.
    """
    # If close_a is a DataFrame with two columns and close_b is None, extract columns
    if close_b is None and isinstance(close_a, pd.DataFrame) and close_a.shape[1] >= 2:
        # Prefer numeric columns
        df = close_a
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            a = df[numeric_cols[0]].values.astype(float)
            b = df[numeric_cols[1]].values.astype(float)
        else:
            # Use first two columns regardless of dtype
            a = df.iloc[:, 0].values.astype(float)
            b = df.iloc[:, 1].values.astype(float)
    else:
        a = _to_1d_array(close_a)
        if close_b is None:
            raise ValueError("close_b must be provided when close_a is not a 2-column DataFrame")
        b = _to_1d_array(close_b)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays/series after extraction")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    # Pre-allocate outputs
    hedge_ratio = np.empty(n, dtype=float)
    hedge_ratio.fill(np.nan)

    spread = np.empty(n, dtype=float)
    spread.fill(np.nan)

    rolling_mean = np.empty(n, dtype=float)
    rolling_mean.fill(np.nan)

    rolling_std = np.empty(n, dtype=float)
    rolling_std.fill(np.nan)

    zscore = np.empty(n, dtype=float)
    zscore.fill(np.nan)

    # Small epsilon to avoid division by zero
    EPS = 1e-8

    # Rolling OLS for hedge ratio (Price_A = beta * Price_B + intercept)
    for i in range(n):
        start = 0 if hedge_lookback is None else max(0, i - hedge_lookback + 1)
        ya = a[start : i + 1]
        xb = b[start : i + 1]

        # Mask out NaNs in the window
        mask = (~np.isnan(ya)) & (~np.isnan(xb))
        if mask.sum() >= 2:
            y_w = ya[mask]
            x_w = xb[mask]
            X = np.column_stack((x_w, np.ones_like(x_w)))
            try:
                coef, *_ = np.linalg.lstsq(X, y_w, rcond=None)
                slope = float(coef[0])
            except Exception:
                slope = np.nan
            if np.isnan(slope):
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 1.0
            else:
                hedge_ratio[i] = slope
        else:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 1.0

        # Compute spread at time i using hedge ratio up to i (causal)
        if np.isnan(a[i]) or np.isnan(b[i]) or np.isnan(hedge_ratio[i]):
            spread[i] = np.nan
        else:
            spread[i] = a[i] - hedge_ratio[i] * b[i]

        # Rolling mean/std for z-score
        zs_start = max(0, i - zscore_lookback + 1)
        window = spread[zs_start : i + 1]
        valid = ~np.isnan(window)
        if valid.sum() >= 1:
            m = float(np.mean(window[valid]))
            s = float(np.std(window[valid], ddof=0))
        else:
            m = 0.0
            s = 0.0

        rolling_mean[i] = m
        rolling_std[i] = s

        # Compute zscore safely
        if s < EPS:
            zscore[i] = 0.0
        else:
            if np.isnan(spread[i]):
                zscore[i] = np.nan
            else:
                zscore[i] = (spread[i] - m) / s

    # Fill any remaining NaNs for hedge_ratio and zscore to ensure no NaNs after warmup
    if np.isnan(hedge_ratio).any():
        hr_series = pd.Series(hedge_ratio)
        hr_series = hr_series.ffill().bfill().fillna(1.0)
        hedge_ratio = hr_series.values.astype(float)

    if np.isnan(zscore).any():
        zs_series = pd.Series(zscore)
        zs_series = zs_series.fillna(0.0)
        zscore = zs_series.values.astype(float)

    return {
        "zscore": zscore,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
    }


def order_func(
    c: Any,
    close_a: Union[np.ndarray, pd.Series, list],
    close_b: Union[np.ndarray, pd.Series, list],
    zscore: Union[np.ndarray, pd.Series, list],
    hedge_ratio: Union[np.ndarray, pd.Series, list],
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function compatible with vectorbt flexible order API wrapper.

    Args:
        c: Order context (must provide attributes i, col, position_now for flexible wrapper)
        close_a, close_b: price arrays
        zscore: precomputed zscore array
        hedge_ratio: precomputed hedge ratio array
        entry_threshold, exit_threshold, stop_threshold: thresholds for trade logic
        notional_per_leg: fixed notional to size Asset A (Asset B scaled by hedge ratio)

    Returns:
        (size, size_type, direction)
    """
    ca = np.asarray(close_a, dtype=float)
    cb = np.asarray(close_b, dtype=float)
    zs = np.asarray(zscore, dtype=float)
    hr = np.asarray(hedge_ratio, dtype=float)

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0

    pos_now = float(getattr(c, "position_now", 0.0)) if hasattr(c, "position_now") else 0.0

    n = len(ca)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(ca[i]) if not np.isnan(ca[i]) else np.nan
    price_b = float(cb[i]) if not np.isnan(cb[i]) else np.nan
    z = float(zs[i]) if not np.isnan(zs[i]) else 0.0
    hedge = float(hr[i]) if not np.isnan(hr[i]) else 1.0

    if price_a <= 0 or np.isnan(price_a):
        units_a = 0.0
    else:
        units_a = float(notional_per_leg) / price_a

    units_b = abs(hedge) * units_a

    def close_position(size_now: float) -> Tuple[float, int, int]:
        if size_now == 0 or np.isnan(size_now):
            return (np.nan, 0, 0)
        if size_now > 0:
            return (abs(size_now), 0, 2)
        else:
            return (abs(size_now), 0, 1)

    z_prev = float(zs[i - 1]) if i > 0 else 0.0
    crossed_zero = (z_prev <= 0 and z >= 0) or (z_prev >= 0 and z <= 0)

    # Stop-loss
    if abs(z) > stop_threshold:
        return close_position(pos_now)

    # If position open, check for exit
    if pos_now != 0.0:
        if crossed_zero:
            return close_position(pos_now)
        return (np.nan, 0, 0)

    # No open position: check for entry
    if col == 0:
        if z > entry_threshold:
            if units_a <= 0 or np.isnan(units_a):
                return (np.nan, 0, 0)
            return (abs(units_a), 0, 2)
        if z < -entry_threshold:
            if units_a <= 0 or np.isnan(units_a):
                return (np.nan, 0, 0)
            return (abs(units_a), 0, 1)

    if col == 1:
        if z > entry_threshold:
            if units_b <= 0 or np.isnan(units_b):
                return (np.nan, 0, 0)
            return (abs(units_b), 0, 1)
        if z < -entry_threshold:
            if units_b <= 0 or np.isnan(units_b):
                return (np.nan, 0, 0)
            return (abs(units_b), 0, 2)

    return (np.nan, 0, 0)
