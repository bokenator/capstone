# Pairs trading strategy implementation for vectorbt
# Exports: compute_spread_indicators, order_func

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _compute_hr_for_pair(y: np.ndarray, x: np.ndarray, hedge_lookback: int) -> np.ndarray:
    """Compute rolling OLS slope (hedge ratio) for a single pair y ~ x.

    Uses past data up to and including t with window size hedge_lookback. When fewer
    than 2 valid observations are available, carries forward previous hedge ratio
    (or uses 1.0 for the first element).
    """
    n = y.shape[0]
    hr = np.empty(n, dtype=float)
    prev_hr = 1.0

    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x_win = x[start : i + 1]
        y_win = y[start : i + 1]

        mask = np.isfinite(x_win) & np.isfinite(y_win)
        x_valid = x_win[mask]
        y_valid = y_win[mask]

        if x_valid.size >= 2:
            try:
                slope, intercept, r_value, p_value, stderr = stats.linregress(x_valid, y_valid)
                if np.isfinite(slope):
                    hr_val = float(slope)
                else:
                    hr_val = prev_hr
            except Exception:
                hr_val = prev_hr
        else:
            hr_val = prev_hr if i > 0 else 1.0

        hr[i] = hr_val
        prev_hr = hr_val

    return hr


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs strategy.

    Supports 1D and 2D inputs. If inputs are 2D with different numbers of columns,
    a single-column input will be broadcast across columns of the other input.

    Args:
        close_a: Array-like of prices for asset A (will be converted to numpy array).
        close_b: Array-like of prices for asset B (will be converted to numpy array).
        hedge_lookback: Lookback (window) for the rolling OLS hedge ratio. If fewer
            observations are available, an OLS is fitted on available history (no
            lookahead).
        zscore_lookback: Lookback for rolling mean/std of the spread used to compute z-score.

    Returns:
        Dict with keys:
            - 'hedge_ratio': numpy array of hedge ratios (shape (n,) or (n, m))
            - 'zscore': numpy array of z-scores (shape (n,) or (n, m))
            - 'spread': numpy array of spread values (shape (n,) or (n, m))
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("Inputs must be 1D or 2D arrays/series")

    # Make both arrays 2D with shape (n, m)
    # Time dimension is axis 0
    if a.ndim == 1:
        a2 = a.reshape(-1, 1)
    else:
        a2 = a

    if b.ndim == 1:
        b2 = b.reshape(-1, 1)
    else:
        b2 = b

    if a2.shape[0] != b2.shape[0]:
        raise ValueError("close_a and close_b must have the same number of rows (time length)")

    # Align columns: if one has 1 column and the other has m, broadcast the single column
    m_a = a2.shape[1]
    m_b = b2.shape[1]

    if m_a == m_b:
        m = m_a
    elif m_a == 1 and m_b > 1:
        a2 = np.repeat(a2, m_b, axis=1)
        m = m_b
    elif m_b == 1 and m_a > 1:
        b2 = np.repeat(b2, m_a, axis=1)
        m = m_a
    else:
        raise ValueError("Number of columns in close_a and close_b must match, or one of them must be 1")

    n = a2.shape[0]

    # Compute hedge_ratio per column pair
    hedge_ratio = np.empty((n, m), dtype=float)
    for col in range(m):
        hedge_ratio[:, col] = _compute_hr_for_pair(a2[:, col], b2[:, col], hedge_lookback)

    # Compute spread elementwise
    spread = a2 - hedge_ratio * b2

    # Rolling mean/std with pandas DataFrame to support 2D
    df_spread = pd.DataFrame(spread)
    roll_mean = df_spread.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    roll_std = df_spread.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Compute zscore safely
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (spread - roll_mean) / roll_std
    invalid_mask = ~np.isfinite(zscore)
    zscore[invalid_mask] = 0.0

    # If only one column, return 1D arrays for convenience
    if spread.shape[1] == 1:
        return {
            "hedge_ratio": hedge_ratio.ravel(),
            "spread": spread.ravel(),
            "zscore": zscore.ravel(),
        }

    # Otherwise return 2D arrays
    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: Any,
    close_b: Any,
    zscore: Any,
    hedge_ratio: Any,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Order function for a flexible two-asset pairs strategy.

    Args:
        c: Context object provided by vectorbt (or simulated in tests). Expected to have
           attributes: i (int, current index) and col (int, 0 for asset A, 1 for asset B),
           and position_now (float) representing current position (in amount units).
        close_a, close_b: Arrays of close prices.
        zscore: Array of z-scores (aligned with prices).
        hedge_ratio: Array of hedge ratios (aligned with prices).
        entry_threshold: Threshold to enter trades (default 2.0).
        exit_threshold: Threshold for mean reversion exit (0.0 by default). Not directly used
                       because we detect crossing zero via sign change.
        stop_threshold: Stop-loss threshold for |zscore| (default 3.0).
        notional_per_leg: Dollar notional to allocate per leg. We convert this to number of units
                          for asset A and scale asset B units by the hedge ratio.

    Returns:
        Tuple (size, size_type, direction) as expected by the wrapper. If no order is issued,
        returns (np.nan, 0, 0) so the wrapper recognizes NoOrder.
    """
    # Protect against bad context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Convert arrays to numpy for indexing
    close_a_arr = np.asarray(close_a)
    close_b_arr = np.asarray(close_b)
    z_arr = np.asarray(zscore)
    hr_arr = np.asarray(hedge_ratio)

    # Normalize prices to 1D arrays
    price_a = np.asarray(close_a_arr).reshape(-1)
    price_b = np.asarray(close_b_arr).reshape(-1)

    # Helper to get element for possibly 1D or 2D arrays
    def _get_elem(arr, idx_i, col_idx):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[idx_i]
        elif arr.ndim == 2:
            if 0 <= col_idx < arr.shape[1]:
                return arr[idx_i, col_idx]
            else:
                return arr[idx_i, 0]
        else:
            return arr.reshape(arr.shape[0], -1)[idx_i, 0]

    # Bounds check
    if i < 0 or i >= z_arr.shape[0]:
        return (np.nan, 0, 0)

    price_a_i = float(price_a[i]) if np.isfinite(price_a[i]) else np.nan
    price_b_i = float(price_b[i]) if np.isfinite(price_b[i]) else np.nan

    # Get z and hr for this column (support 1D/2D)
    try:
        z_val = _get_elem(z_arr, i, col)
        z = float(z_val) if np.isfinite(z_val) else np.nan
    except Exception:
        z = np.nan

    try:
        hr_val = _get_elem(hr_arr, i, col)
        hr = float(hr_val) if np.isfinite(hr_val) else 0.0
    except Exception:
        hr = 0.0

    # Current position for this column (in amount units). If missing, assume 0.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # SizeType and Direction enums (by integer value in vectorbt)
    SIZE_TYPE_AMOUNT = 0  # Amount (units)
    SIZE_TYPE_TARGET_AMOUNT = 3  # TargetAmount
    DIRECTION_LONG = 0  # LongOnly
    DIRECTION_SHORT = 1  # ShortOnly
    DIRECTION_BOTH = 2  # Both

    # Helper to return NoOrder
    def no_order():
        return (float("nan"), SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # If price NaN or z NaN -> no order
    if not np.isfinite(price_a_i) or not np.isfinite(price_b_i) or not np.isfinite(z):
        return no_order()

    # Compute unit sizing based on notional per leg (units of A)
    units_a = 0.0
    if price_a_i > 0:
        units_a = float(notional_per_leg / price_a_i)

    # Units for asset B scale by absolute hedge ratio to get magnitude; sign is handled via direction
    units_b = float(abs(hr) * units_a)

    # Determine if an exit condition holds (stop-loss or cross zero)
    exit_now = False
    # Stop-loss
    if abs(z) > float(stop_threshold):
        exit_now = True

    # Cross zero (sign change) - check previous value if available
    prev_z = None
    try:
        prev_z = _get_elem(z_arr, i - 1, col) if i > 0 else None
        if prev_z is not None and np.isfinite(prev_z):
            if prev_z * z < 0:
                exit_now = True
    except Exception:
        pass

    # If there is a position now and exit condition -> close this asset's position
    eps = 1e-12
    if exit_now and abs(pos_now) > eps:
        # Use TargetAmount with target 0 and Both direction to close regardless of current sign
        return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # Entry logic: only enter if currently flat
    if abs(pos_now) <= eps:
        # Long A / Short B when z < -entry_threshold
        if z < -float(entry_threshold):
            if col == 0:
                # Buy asset A (units_a)
                size = units_a if units_a > 0 else 0.0
                if size <= 0:
                    return no_order()
                return (size, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            else:
                # Short asset B (units_b)
                size = units_b if units_b > 0 else 0.0
                if size <= 0:
                    return no_order()
                return (size, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)

        # Short A / Long B when z > entry_threshold
        if z > float(entry_threshold):
            if col == 0:
                # Short asset A (units_a)
                size = units_a if units_a > 0 else 0.0
                if size <= 0:
                    return no_order()
                return (size, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
            else:
                # Long asset B (units_b)
                size = units_b if units_b > 0 else 0.0
                if size <= 0:
                    return no_order()
                return (size, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # Otherwise, no order
    return no_order()
