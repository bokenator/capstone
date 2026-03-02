# Pairs trading strategy implementation for vectorbt backtests
# Exports:
# - compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
# - order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> tuple

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread for a pair of assets.

    Args:
        close_a (np.ndarray): Prices for asset A (1D array-like).
        close_b (np.ndarray): Prices for asset B (1D array-like).
        hedge_lookback (int): Lookback window for rolling OLS (number of bars).
        zscore_lookback (int): Lookback window for rolling mean/std of the spread.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing at least:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-score values (same length as inputs)
            - 'spread': np.ndarray of spread values (same length as inputs)

    Notes:
        - Uses only historical data up to and including the current index (no lookahead).
        - Handles NaNs by dropping them inside each rolling window.
        - For very small windows, uses a sensible fallback (slope=0 when insufficient data).
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Compute rolling OLS (slope) for hedge ratio using past window up to current index
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x_win = b[start : i + 1]
        y_win = a[start : i + 1]

        # Remove NaNs pairwise
        mask = ~np.isnan(x_win) & ~np.isnan(y_win)
        x = x_win[mask]
        y = y_win[mask]

        slope = 0.0
        if x.size >= 2:
            # Solve y = slope * x + intercept via least squares
            X = np.vstack([x, np.ones_like(x)]).T
            try:
                # lstsq returns coef [slope, intercept]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                slope = float(coef[0])
                if not np.isfinite(slope):
                    slope = 0.0
            except Exception:
                slope = 0.0
        elif x.size == 1:
            # Fallback: use ratio if possible
            if x[0] != 0 and np.isfinite(x[0]) and np.isfinite(y[0]):
                slope = float(y[0] / x[0])
            else:
                slope = 0.0
        else:
            slope = 0.0

        hedge_ratio[i] = slope
        # Compute spread at current index (use available slope). If price is nan keep spread as nan
        if np.isnan(a[i]) or np.isnan(b[i]):
            spread[i] = np.nan
        else:
            spread[i] = a[i] - slope * b[i]

    # Compute rolling mean/std for spread and z-score (use only past values up to current index)
    for i in range(n):
        start = max(0, i - zscore_lookback + 1)
        window = spread[start : i + 1]
        # Exclude NaNs inside the window
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            mean_w = 0.0
            std_w = 0.0
        else:
            mean_w = float(np.mean(valid))
            # Population std (ddof=0) - avoids NaN for single sample
            std_w = float(np.std(valid, ddof=0))

        if std_w == 0.0 or np.isnan(std_w):
            zscore[i] = 0.0
        else:
            if np.isnan(spread[i]):
                zscore[i] = np.nan
            else:
                zscore[i] = float((spread[i] - mean_w) / std_w)

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
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
    Order function for flexible vectorbt backtesting (pairs trading).

    Returns a tuple (size, size_type, direction) or (np.nan, 0, 0) for NoOrder.

    - size: positive magnitude of the order (in number of shares)
    - size_type: 0 -> absolute number of shares
    - direction: 1 -> buy (long), 2 -> sell (short)

    Logic:
    - Entry when zscore > entry_threshold (short A, long B) or zscore < -entry_threshold (long A, short B)
    - Exit when zscore crosses zero (sign change) or |zscore| > stop_threshold (stop-loss)
    - Position sizing: base_qty = notional_per_leg / price_A; Asset B sized as hedge_ratio * base_qty

    Note: The function only uses historical data up to index c.i (no lookahead).
    """
    # Defensive retrieval of necessary attributes
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Get current position for this column (number of shares). If not available, assume 0.
    pos_now = getattr(c, "position_now", None)
    if pos_now is None:
        # Try alternative attribute names that vectorbt may expose
        pos_now = getattr(c, "position", None)
    if pos_now is None:
        # Fallback to 0.0
        pos_now = 0.0

    try:
        pA = float(np.asarray(close_a)[i])
        pB = float(np.asarray(close_b)[i])
    except Exception:
        return (np.nan, 0, 0)

    # Safely read indicators
    z = float(zscore[i]) if i >= 0 and i < len(zscore) and not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if i >= 0 and i < len(hedge_ratio) and not np.isnan(hedge_ratio[i]) else np.nan

    # If essential values are NaN, don't trade
    if np.isnan(z) or np.isnan(hr) or np.isnan(pA) or np.isnan(pB):
        return (np.nan, 0, 0)

    # Determine whether to exit due to stop loss
    if abs(z) > float(stop_threshold):
        target_a = 0.0
        target_b = 0.0
    else:
        # Detect crossing of z through zero (compare with previous bar)
        z_prev = None
        if i > 0:
            zp = zscore[i - 1]
            if not np.isnan(zp):
                z_prev = float(zp)

        crossed_zero = False
        if z_prev is not None:
            # Consider crossing if sign changed (zero is treated as crossing)
            if (z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0):
                crossed_zero = True

        if crossed_zero:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short A, Long B
                if pA == 0:
                    return (np.nan, 0, 0)
                base_qty = float(notional_per_leg) / float(pA)
                target_a = -base_qty
                target_b = float(base_qty) * float(hr)
            elif z < -entry_threshold:
                # Long A, Short B
                if pA == 0:
                    return (np.nan, 0, 0)
                base_qty = float(notional_per_leg) / float(pA)
                target_a = base_qty
                target_b = -float(base_qty) * float(hr)
            else:
                # No signal: keep current positions
                # We don't have full portfolio state here for the other column,
                # but for this column we don't want to change position
                # so set target equal to current pos
                if col == 0:
                    return (np.nan, 0, 0)
                else:
                    return (np.nan, 0, 0)

    # At this point we have target_a and target_b
    if col == 0:
        target = float(target_a)
    else:
        target = float(target_b)

    # Compute order size as difference between target and current holdings
    size_diff = float(target) - float(pos_now)

    # If there is nothing to do, return NoOrder
    if np.isclose(size_diff, 0.0, atol=1e-12):
        return (np.nan, 0, 0)

    size_abs = float(abs(size_diff))

    # Size type: 0 -> absolute number of shares
    size_type = 0
    # Direction: 1 -> buy (long), 2 -> sell (short)
    direction = 1 if size_diff > 0 else 2

    return (size_abs, int(size_type), int(direction))
