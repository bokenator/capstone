# Pairs trading indicator and order function implementation (robust input handling)
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Global state to avoid returning the same (bar,col) order multiple times
_LAST_ORDER_BAR: int | None = None
_SUBMITTED_ORDERS: set[tuple[int, int]] = set()


def _to_1d_array(x: Any) -> np.ndarray:
    """
    Convert various input types to a 1D numpy array (prefers the first meaningful column
    when given a 2D structure).
    """
    # If it's a pandas Series
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)

    # If it's a pandas DataFrame, pick the first numeric column
    if isinstance(x, pd.DataFrame):
        # Prefer a column named 'close' if present
        if 'close' in x.columns:
            s = x['close']
            return s.to_numpy(dtype=float)
        # Otherwise pick the first column
        for col in x.columns:
            try:
                arr = x[col].to_numpy(dtype=float)
                return arr
            except Exception:
                continue
        # If nothing worked, fallback to numpy array flattening
        arr = x.to_numpy(dtype=float)
        if arr.ndim == 1:
            return arr
        # Take first column
        return arr[:, 0].astype(float)

    # If it's a numpy array
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.array([float(x)])
        if x.ndim == 1:
            return x.astype(float)
        # If 2D or higher, take first column
        return x[:, 0].astype(float)

    # If it's a list/tuple
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim >= 2:
            return arr[:, 0].astype(float)

    # As last resort, try to convert directly
    return np.asarray(x, dtype=float).reshape(-1)


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score for a pair of assets.

    This function is intentionally flexible in input types: it accepts numpy arrays,
    pandas Series/DataFrame, lists, and will attempt to extract 1D price series for
    both assets. When a 2D object is provided, the first numeric column is used.

    Args:
        close_a: Prices for asset A (array-like, Series, or DataFrame)
        close_b: Prices for asset B (array-like, Series, DataFrame, or column name if close_a is DataFrame)
        hedge_lookback: Window size for rolling OLS regression (uses available data when fewer bars)
        zscore_lookback: Window size for rolling mean/std of spread

    Returns:
        Dict with keys:
            - "zscore": z-score array (np.ndarray, same length as the paired inputs)
            - "hedge_ratio": hedge ratio per bar (np.ndarray, same length)

    Notes:
        - Uses only past data up to and including the current bar (no lookahead).
        - For very early bars where not enough data exists for a full window, regression
          is run on the available data (min 2 points). For the first bar a default
          hedge ratio of 1.0 is used.
    """
    # If close_a is a DataFrame and close_b is a string column name, extract columns
    if isinstance(close_a, pd.DataFrame) and isinstance(close_b, str) and close_b in close_a.columns:
        # If the DataFrame has exactly 2 columns, use them as A and B
        if close_a.shape[1] == 2:
            cols = list(close_a.columns)
            if cols[0] == close_b and cols[1] != close_b:
                ca = close_a[cols[1]]
                cb = close_a[cols[0]]
            else:
                ca = close_a[cols[0]]
                cb = close_a[cols[1]] if cols[1] != close_b else close_a[cols[0]]
            a = _to_1d_array(ca)
            b = _to_1d_array(cb)
        else:
            # Prefer a column named 'asset_a' or the first column not equal to close_b
            cols = list(close_a.columns)
            for col in cols:
                if col != close_b:
                    a = _to_1d_array(close_a[col])
                    break
            else:
                a = _to_1d_array(close_a.iloc[:, 0])
            b = _to_1d_array(close_a[close_b])
    else:
        # General conversion to 1D arrays
        a = _to_1d_array(close_a)
        b = _to_1d_array(close_b)

    # Truncate to the minimum common length to support truncated inputs used in tests
    min_len = min(a.shape[0], b.shape[0])
    if min_len == 0:
        # Return empty arrays
        return {"zscore": np.array([], dtype=float), "hedge_ratio": np.array([], dtype=float)}

    if a.shape[0] != b.shape[0]:
        a = a[:min_len]
        b = b[:min_len]

    n = len(a)

    # Prepare hedge ratio array
    hedge_ratio = np.empty(n, dtype=float)
    hedge_ratio.fill(np.nan)

    # Compute rolling OLS slope (hedge ratio) using past data only
    # Regression: a = beta * b + intercept; hedge_ratio = beta
    for t in range(n):
        # Window is inclusive of current bar
        start = max(0, t - hedge_lookback + 1)
        x = b[start : t + 1]
        y = a[start : t + 1]

        if x.size >= 2:
            # Design matrix with intercept
            X = np.column_stack((x, np.ones(x.size)))
            # Solve least squares
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                beta = float(coef[0])
            except Exception:
                # Fallback to previous hedge ratio or 1.0
                beta = hedge_ratio[t - 1] if t > 0 and not np.isnan(hedge_ratio[t - 1]) else 1.0
        else:
            # Not enough points: use previous estimate or default 1.0
            beta = hedge_ratio[t - 1] if t > 0 and not np.isnan(hedge_ratio[t - 1]) else 1.0

        hedge_ratio[t] = beta

    # Compute spread using hedge ratio (elementwise)
    spread = a - hedge_ratio * b

    # Rolling mean and std for z-score (past data only)
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=2).mean().to_numpy()
    # Use population std (ddof=0)
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=2).std(ddof=0).to_numpy()

    # Avoid division by zero
    eps = 1e-8
    safe_std = np.where(np.isnan(roll_std) | (roll_std < eps), eps, roll_std)

    zscore = (spread - roll_mean) / safe_std

    # Ensure outputs are numpy arrays of correct length
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)
    zscore = np.asarray(zscore, dtype=float)

    return {
        "zscore": zscore,
        "hedge_ratio": hedge_ratio,
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
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading (flexible multi-asset mode).

    Signature expected by the backtest wrapper:
        order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold)

    The context `c` is expected to have attributes:
        - i: current integer bar index
        - col: which asset this call is for (0 for asset A, 1 for asset B)
        - position_now: current position size (in units) for this asset (0 if flat)
        - cash_now (optional)

    Returns a tuple (size, size_type, direction) where:
        - size_type = 0 -> size is interpreted as number of units
        - direction = 1 -> buy/long, -1 -> sell/short
        - To indicate no order, return (np.nan, 0, 0)
    """
    global _LAST_ORDER_BAR, _SUBMITTED_ORDERS

    # Constants for order encoding
    SIZE_TYPE_UNITS = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = -1

    # Position sizing parameters
    NOTIONAL_PER_LEG = 10_000.0

    # Extract context info
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0) if getattr(c, "position_now", None) is not None else 0.0)

    # Reset per-bar tracking when bar changes
    if _LAST_ORDER_BAR != i:
        _LAST_ORDER_BAR = i
        _SUBMITTED_ORDERS.clear()

    # If we've already submitted an order for this (bar, col), do not return it again
    if (i, col) in _SUBMITTED_ORDERS:
        return (float("nan"), SIZE_TYPE_UNITS, 0)

    # Safe indexing
    if i < 0 or i >= len(zscore):
        return (float("nan"), SIZE_TYPE_UNITS, 0)

    # Current indicator values
    z = float(zscore[i])
    # Previous z for crossing detection
    z_prev = float(zscore[i - 1]) if i > 0 else float("nan")

    # Hedge ratio at this bar, fallback to 1.0 if invalid
    h = float(hedge_ratio[i]) if i < len(hedge_ratio) and np.isfinite(hedge_ratio[i]) else 1.0

    # Prices at current bar
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan

    # Default: no actionable target
    target_a = None
    target_b = None

    # Stop loss has highest priority
    if np.isfinite(z) and abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Exit on crossing zero
        crossed = False
        if np.isfinite(z_prev) and np.isfinite(z):
            if (z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0):
                crossed = True

        if crossed or (np.isfinite(z) and abs(z) <= exit_threshold):
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry conditions
            if np.isfinite(z) and z > entry_threshold:
                # Short A, Long B
                units_a = NOTIONAL_PER_LEG / price_a if price_a > 0 else 0.0
                units_b = abs(h) * units_a
                target_a = -units_a
                target_b = +units_b
            elif np.isfinite(z) and z < -entry_threshold:
                # Long A, Short B
                units_a = NOTIONAL_PER_LEG / price_a if price_a > 0 else 0.0
                units_b = abs(h) * units_a
                target_a = +units_a
                target_b = -units_b
            else:
                target_a = None
                target_b = None

    # Determine which asset this call is for and compute required delta (in units)
    if col == 0:
        if target_a is None:
            return (float("nan"), SIZE_TYPE_UNITS, 0)
        target = float(target_a)
    else:
        if target_b is None:
            return (float("nan"), SIZE_TYPE_UNITS, 0)
        target = float(target_b)

    delta = target - pos_now

    # No change
    if abs(delta) < 1e-8:
        return (float("nan"), SIZE_TYPE_UNITS, 0)

    size = abs(delta)
    direction = DIRECTION_LONG if delta > 0 else DIRECTION_SHORT

    # Mark this (bar, col) as submitted so we don't return duplicate orders
    _SUBMITTED_ORDERS.add((i, col))

    return (float(size), SIZE_TYPE_UNITS, int(direction))
