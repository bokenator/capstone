from typing import Any, Dict, Tuple

import numpy as np
import scipy


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    Args:
        close_a: Array of close prices for asset A.
        close_b: Array of close prices for asset B.
        hedge_lookback: Lookback (in bars) for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback (in bars) for rolling mean/std of the spread.

    Returns:
        A dict with keys:
            - "hedge_ratio": np.ndarray of hedge ratios (same length as inputs)
            - "zscore": np.ndarray of z-score of the spread (same length as inputs)

    Notes:
        - Uses scipy.stats.linregress for OLS on each rolling window.
        - Warm-up periods are filled with np.nan.
        - Handles NaN values in input by skipping windows with insufficient data.
    """
    a = np.array(close_a, dtype=np.float64)
    b = np.array(close_b, dtype=np.float64)

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=np.float64)

    # Rolling OLS to estimate hedge ratio (slope of A ~ hedge_ratio * B)
    for i in range(n):
        if i >= hedge_lookback - 1:
            start = i - hedge_lookback + 1
            x_window = b[start : i + 1]
            y_window = a[start : i + 1]

            mask = np.isfinite(x_window) & np.isfinite(y_window)
            if np.sum(mask) >= 2:
                # Use fully-qualified scipy.stats.linregress
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                    x_window[mask], y_window[mask]
                )
                hedge_ratio[i] = float(slope)
            else:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if np.isfinite(hedge_ratio[i]) and np.isfinite(a[i]) and np.isfinite(b[i]):
            spread[i] = a[i] - hedge_ratio[i] * b[i]
        else:
            spread[i] = np.nan

    # Rolling mean and std of spread for z-score
    rolling_mean = np.full(n, np.nan, dtype=np.float64)
    rolling_std = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i >= zscore_lookback - 1:
            start = i - zscore_lookback + 1
            window = spread[start : i + 1]
            mask = np.isfinite(window)
            if np.sum(mask) >= 1:
                vals = window[mask]
                rolling_mean[i] = np.mean(vals)
                rolling_std[i] = np.std(vals)
            else:
                rolling_mean[i] = np.nan
                rolling_std[i] = np.nan
        else:
            rolling_mean[i] = np.nan
            rolling_std[i] = np.nan

    zscore = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if np.isfinite(spread[i]) and np.isfinite(rolling_mean[i]) and np.isfinite(rolling_std[i]) and rolling_std[i] > 0:
            zscore[i] = (spread[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = np.nan

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading strategy.

    Signals (per the provided spec):
      - Entry: zscore > entry_threshold  -> Short A, Long B
               zscore < -entry_threshold -> Long A, Short B
      - Exit: zscore crosses 0.0 -> close positions
      - Stop-loss: |zscore| > stop_threshold -> close positions

    Position sizing:
      - Fixed notional per leg: $10,000 (size_type indicated as 'value' = 1)
      - Returns a tuple (size, size_type, direction). If no order, returns (np.nan, 0, 0).

    Notes:
      - The function is called for each asset (col 0 = asset A, col 1 = asset B) via a wrapper.
      - `c` is a context-like object with attributes i (index), col (0 or 1), position_now (current position), cash_now.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Current position for this asset (can be 0 if none)
    position_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])
    current_z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    NOTIONAL_PER_LEG = 10000.0
    # We use size_type = 1 to indicate "value" / notional sizing
    SIZE_TYPE_NOTIONAL = 1

    NO_ORDER: Tuple[float, int, int] = (np.nan, 0, 0)

    # If price or zscore is invalid, do nothing
    if not np.isfinite(price) or not np.isfinite(current_z):
        return NO_ORDER

    in_position = np.abs(position_now) > 1e-12

    # In vectorbt the direction codes are integers; use 1 for LONG (buy) and 2 for SHORT (sell)
    DIR_LONG = 1
    DIR_SHORT = 2

    # Stop-loss: close if |zscore| > stop_threshold
    if in_position and np.isfinite(current_z) and np.abs(current_z) > stop_threshold:
        # Close by issuing an opposite direction order of the same notional
        if position_now > 0:
            direction = DIR_SHORT
        elif position_now < 0:
            direction = DIR_LONG
        else:
            return NO_ORDER
        return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, int(direction))

    # Exit on z-score crossing zero (sign change)
    if in_position and np.isfinite(prev_z) and (prev_z * current_z < 0):
        if position_now > 0:
            direction = DIR_SHORT
        elif position_now < 0:
            direction = DIR_LONG
        else:
            return NO_ORDER
        return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, int(direction))

    # Entry logic: only enter when not in position
    if not in_position:
        if current_z > entry_threshold:
            # Short A, Long B
            if col == 0:
                return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, DIR_SHORT)
            else:
                return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, DIR_LONG)
        elif current_z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, DIR_LONG)
            else:
                return (NOTIONAL_PER_LEG, SIZE_TYPE_NOTIONAL, DIR_SHORT)

    # Otherwise, no order
    return NO_ORDER