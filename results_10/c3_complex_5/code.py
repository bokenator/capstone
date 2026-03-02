"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> tuple

Notes:
- Rolling OLS hedge ratio computed using only past data (expanding/rolling with available samples).
- Rolling z-score computed using rolling mean/std with available samples (no NaNs after warmup if data present).
- order_func returns (size, size_type, direction). size_type=0 (absolute size), direction: 1=buy, 2=sell. Return size=np.nan to indicate no order.

CRITICAL: Do not use numba in this module.
"""

from typing import Any, Dict, Tuple

import numpy as np


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS slope), spread, rolling mean/std and z-score.

    All calculations use only information up to the current time index (no lookahead).
    For early times where full lookback is not available, the available history is used
    (i.e. expanding/variable-length windows).

    Args:
        close_a: Prices for asset A (1D array-like).
        close_b: Prices for asset B (1D array-like).
        hedge_lookback: Lookback for rolling OLS (number of bars). Uses available data if fewer bars exist.
        zscore_lookback: Lookback for rolling mean/std of spread. Uses available data if fewer bars exist.

    Returns:
        Dict with keys:
            - 'hedge_ratio': numpy array of hedge ratios (slope) for each time index
            - 'spread': spread time series (A - hedge_ratio * B)
            - 'rolling_mean': rolling mean of spread
            - 'rolling_std': rolling std of spread
            - 'zscore': z-score of spread
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Prepare output arrays
    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    rolling_mean = np.zeros(n, dtype=float)
    rolling_std = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Small epsilon to avoid division by zero
    EPS = 1e-8

    for i in range(n):
        # Rolling window for hedge regression: use available data up to current index
        start_hr = max(0, i - hedge_lookback + 1)
        a_win = a[start_hr : i + 1]
        b_win = b[start_hr : i + 1]

        # Compute slope (beta) for regression A = alpha + beta * B using OLS (closed-form)
        # beta = cov(A,B) / var(B)
        if b_win.size == 0:
            hr = 0.0
        else:
            mean_a = np.mean(a_win)
            mean_b = np.mean(b_win)
            num = np.sum((a_win - mean_a) * (b_win - mean_b))
            den = np.sum((b_win - mean_b) ** 2)
            if den <= EPS:
                hr = 0.0
            else:
                hr = num / den

        hedge_ratio[i] = hr

        # Spread at time i
        spread_i = a[i] - hr * b[i]
        spread[i] = spread_i

        # Rolling mean/std for spread using zscore_lookback (use available samples)
        start_z = max(0, i - zscore_lookback + 1)
        s_win = spread[start_z : i + 1]
        m = np.mean(s_win)
        std = np.std(s_win)  # population std (ddof=0)
        if std <= EPS:
            std = EPS

        rolling_mean[i] = m
        rolling_std[i] = std

        # Z-score
        zscore[i] = (spread_i - m) / std

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
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
    Order function for flexible multi-asset mode.

    This function computes the desired position (in units) for the current asset (column)
    based on the z-score, hedge ratio and fixed notional per leg. It returns an absolute
    trade size (number of units to trade), size_type=0 (absolute size), and direction
    (1=buy, 2=sell). Returning size=np.nan indicates no order.

    The function uses only information up to the current index c.i (no lookahead).

    Args:
        c: Context object providing attributes:
            - i: current integer bar index
            - col: column index (0 for asset A, 1 for asset B)
            - position_now: current position in units for this column (can be 0.0)
            - cash_now/value_now: current cash or portfolio value (not used but read-friendly)
        close_a/b: arrays of close prices
        zscore: z-score array computed by compute_spread_indicators
        hedge_ratio: hedge ratio array computed by compute_spread_indicators
        entry_threshold: threshold to enter trades (e.g., 2.0)
        exit_threshold: threshold to exit trades (e.g., 0.0) - crossing logic is used
        stop_threshold: stop-loss threshold (e.g., 3.0)
        notional_per_leg: fixed dollar exposure per leg (e.g., 10000.0)

    Returns:
        Tuple of (size, size_type, direction) where size_type=0 (absolute units)
        and direction in {1 (buy), 2 (sell)}. Use size=np.nan to signal no order.
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))

    # Safeguard indices
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current position in units for this asset
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Current indicator values (use only up to i)
    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # Compute base unit sizes using fixed notional per leg
    # units for asset A such that notional in A is approximately notional_per_leg
    # If price is zero (shouldn't happen), fall back to 0 units
    units_a = notional_per_leg / price_a if price_a != 0.0 else 0.0
    # Asset B units should be scaled by hedge ratio (hedge_ratio * units_a)
    units_b = hr * units_a

    # Desired positions in units for both assets
    desired_a = 0.0
    desired_b = 0.0

    # Stop-loss override: if absolute zscore exceeds stop threshold, close positions
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entry rules
        if z > entry_threshold:
            # Short A, Long B
            desired_a = -units_a
            desired_b = units_b
        elif z < -entry_threshold:
            # Long A, Short B
            desired_a = units_a
            desired_b = -units_b
        else:
            # No entry: check for exit on mean reversion (z-score crossing zero)
            prev_z = float(zscore[i - 1]) if i > 0 else np.nan
            crossed_zero = False
            if not np.isnan(prev_z):
                # Detect sign change across zero
                if prev_z == 0.0 and z == 0.0:
                    crossed_zero = True
                else:
                    crossed_zero = (prev_z * z) < 0.0

            # If we are currently in a position for this asset and crossing zero, close
            if pos_now != 0.0 and crossed_zero:
                desired_a = 0.0
                desired_b = 0.0
            else:
                # Maintain existing positions (no order) -> set desired to current position
                # For the current column we set desired to pos_now so the delta is zero.
                if col == 0:
                    desired_a = pos_now
                    # For asset B we cannot read its current position here; order_func is
                    # called separately for each column. For the current column we simply
                    # avoid creating an order by setting desired to pos_now.
                else:
                    desired_b = pos_now

    # Choose desired position for this column
    if col == 0:
        desired = desired_a
    else:
        desired = desired_b

    # If desired is still None-like (e.g., 0.0 default), keep current position
    # Compute delta (units to trade) as desired - current
    delta = desired - pos_now

    # If delta is effectively zero, do not place order
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Determine direction and absolute size
    if delta > 0:
        direction = 1  # Buy
        size = float(delta)
    else:
        direction = 2  # Sell
        size = float(-delta)

    size_type = 0  # absolute size (units)
    return (size, size_type, direction)
