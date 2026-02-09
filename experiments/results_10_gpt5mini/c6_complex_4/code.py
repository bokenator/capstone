import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union


def _to_1d_array(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """Convert input to a 1D numpy float array.

    Accepts numpy arrays, pandas Series, or single-column DataFrames. Raises on multi-column inputs.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # If single column, squeeze
        if arr.shape[1] == 1:
            return arr[:, 0]
        # If single row, squeeze
        if arr.shape[0] == 1:
            return arr[0, :]
        # Otherwise ambiguous
        raise ValueError("Input must be 1D or a single-column 2D array / DataFrame")
    # For higher dims, flatten
    return arr.ravel()


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    close_b: Union[np.ndarray, pd.Series, pd.DataFrame],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS, no-intercept), spread and z-score for a pairs strategy.

    Args:
        close_a: Prices of asset A (array-like)
        close_b: Prices of asset B (array-like)
        hedge_lookback: Lookback window for rolling OLS (uses smaller window at the beginning)
        zscore_lookback: Lookback window for rolling mean/std of the spread

    Returns:
        Dict with keys:
            - 'hedge_ratio': ndarray of hedge ratios (same length as inputs)
            - 'spread': ndarray of spreads (Price_A - hedge_ratio * Price_B)
            - 'rolling_mean': ndarray of rolling mean of spread
            - 'rolling_std': ndarray of rolling std of spread
            - 'zscore': ndarray of z-score of spread

    Notes:
        - The rolling OLS uses an expanding window at the start (min periods = 1) up to hedge_lookback.
        - No lookahead: all rolling calculations use data up to and including the current index.
    """
    # Convert inputs to 1D numpy arrays of float
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no intercept): slope = sum(a*b) / sum(b*b)
    # Use smaller window at the start (min_periods = 1) to avoid long NaN warmup.
    eps = 1e-12
    for i in range(n):
        window_size = min(hedge_lookback, i + 1)
        start = i - window_size + 1
        a_win = a[start : i + 1]
        b_win = b[start : i + 1]

        # If any NaNs in window, skip (produce NaN hedge_ratio)
        if np.isnan(a_win).any() or np.isnan(b_win).any():
            hedge_ratio[i] = np.nan
            spread[i] = np.nan
            continue

        denom = float(np.dot(b_win, b_win))
        if denom <= eps:
            hedge_ratio[i] = np.nan
            spread[i] = np.nan
        else:
            h = float(np.dot(a_win, b_win) / denom)
            hedge_ratio[i] = h
            spread[i] = float(a[i] - h * b[i])

    # Rolling mean and std of spread for z-score (no lookahead)
    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to be consistent and stable
    rolling_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    rm = rolling_mean.to_numpy(dtype=float)
    rs = rolling_std.to_numpy(dtype=float)

    for i in range(n):
        if np.isfinite(spread[i]) and np.isfinite(rm[i]) and np.isfinite(rs[i]) and rs[i] > eps:
            zscore[i] = (spread[i] - rm[i]) / rs[i]
        else:
            zscore[i] = np.nan

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": rm,
        "rolling_std": rs,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    close_b: Union[np.ndarray, pd.Series, pd.DataFrame],
    zscore: Union[np.ndarray, pd.Series],
    hedge_ratio: Union[np.ndarray, pd.Series],
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible two-asset (pairs) trading in vectorbt.

    This function returns a tuple (size, size_type, direction) where:
        - size_type == 0 (Amount) -> size is number of asset units (shares)
        - direction: 0 for long (buy), 1 for short (sell)

    Logic:
        - Entry: when zscore > entry_threshold -> short A, long B (sizes scaled by hedge_ratio)
                 when zscore < -entry_threshold -> long A, short B
        - Exit: when zscore crosses 0 (sign change) -> close both
        - Stop-loss: when |zscore| > stop_threshold -> close both

    The position sizing uses a fixed notional for asset A: notional_per_leg dollars -> shares_A = notional_per_leg / price_A
    Asset B shares are scaled by hedge_ratio: shares_B = hedge_ratio * shares_A (with opposite sign to A)

    Returns (np.nan, 0, 0) to indicate no order for the current column.
    """
    # Ensure arrays (1D)
    close_a_arr = _to_1d_array(close_a)
    close_b_arr = _to_1d_array(close_b)
    z_arr = _to_1d_array(zscore)
    h_arr = _to_1d_array(hedge_ratio)

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    position_now = float(getattr(c, "position_now", 0.0))

    n = len(z_arr)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a_arr[i])
    price_b = float(close_b_arr[i])

    # Basic validation
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Base number of shares for asset A given fixed notional
    shares_a = float(notional_per_leg / price_a)

    z_now = float(z_arr[i]) if np.isfinite(z_arr[i]) else np.nan
    h_now = float(h_arr[i]) if np.isfinite(h_arr[i]) else np.nan

    # Determine events
    prev_z = float(z_arr[i - 1]) if i > 0 and np.isfinite(z_arr[i - 1]) else np.nan
    crossed_zero = False
    if np.isfinite(z_now) and np.isfinite(prev_z):
        crossed_zero = (z_now == 0.0) or (prev_z == 0.0) or (z_now * prev_z < 0.0)

    stop_loss = np.isfinite(z_now) and (abs(z_now) > stop_threshold)

    # Default: hold existing position for the given column
    desired = float(position_now)

    # If stop-loss or crossing zero -> close both legs
    if stop_loss or crossed_zero:
        desired = 0.0
    else:
        # Entry conditions
        if np.isfinite(z_now) and z_now > entry_threshold:
            # Short A, Long B
            if col == 0:
                desired = -shares_a
            else:
                if not np.isfinite(h_now):
                    return (np.nan, 0, 0)
                desired = float(h_now * shares_a)
        elif np.isfinite(z_now) and z_now < -entry_threshold:
            # Long A, Short B
            if col == 0:
                desired = float(shares_a)
            else:
                if not np.isfinite(h_now):
                    return (np.nan, 0, 0)
                desired = float(-h_now * shares_a)
        else:
            # No change: hold
            desired = float(position_now)

    # Calculate trade delta (in asset units)
    delta = desired - float(position_now)

    # If delta is effectively zero, do not place an order
    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    size_type_amount = 0  # Amount (number of units)

    if delta > 0:
        # Need to buy (long)
        size = float(delta)
        direction = 0  # Long
    else:
        size = float(-delta)
        direction = 1  # Short (sell)

    return (size, size_type_amount, direction)