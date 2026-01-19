"""
Pairs trading strategy: compute rolling hedge ratio via OLS, compute spread and z-score,
and provide an order function for vectorbt flexible from_order_func (use_numba=False).

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold) -> tuple[float, int, int]

Notes:
- No numba usage.
- Uses scipy.stats.linregress for rolling OLS.
- Position sizing: fixed notional $10,000 per leg (we use A as reference and scale B by hedge_ratio).

"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Simple module-level cooldown tracker to avoid excessive churn of orders
_LAST_TRADE_IDX = [-1_000_000, -1_000_000]
_COOLDOWN_BARS = 5  # minimum bars to wait before re-entering on the same asset

# Safeguard to avoid emitting huge number of orders (prevents hitting vectorbt internal limit)
_MAX_ORDERS = 1000
_ORDERS_EMITTED = 0


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread, and z-score for a pair of assets.

    Args:
        close_a: 1D numpy array of closes for asset A.
        close_b: 1D numpy array of closes for asset B.
        hedge_lookback: Window length for rolling OLS to estimate hedge ratio.
        zscore_lookback: Window length for rolling mean/std of the spread to compute z-score.

    Returns:
        A dict containing numpy arrays: 'hedge_ratio', 'spread', 'zscore'.

    Notes:
        - If there are insufficient observations for the rolling windows or NaNs in the
          regression window, results will be NaN for those timestamps.
    """
    # Validate inputs
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    # Rolling OLS regression: regress A on B in each rolling window to get slope (hedge ratio)
    # Use a simple loop for clarity and robust NaN handling.
    for i in range(n):
        if i + 1 >= hedge_lookback:
            start = i + 1 - hedge_lookback
            a_win = close_a[start : i + 1]
            b_win = close_b[start : i + 1]

            # If there are NaNs in the window, skip
            if np.isnan(a_win).any() or np.isnan(b_win).any():
                hedge_ratio[i] = np.nan
                continue

            # If B is constant, slope is undefined -> linregress will produce nan
            try:
                lr = stats.linregress(b_win, a_win)
                slope = float(lr.slope) if np.isfinite(lr.slope) else np.nan
            except Exception:
                slope = np.nan

            hedge_ratio[i] = slope

    # Compute spread. If hedge_ratio is NaN, spread is NaN as well.
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std of spread for z-score
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Compute z-score safely
    zscore = (spread_series - roll_mean) / roll_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

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
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible from_order_func (no numba).

    This function is called once per asset (simulated by wrapper) and must return a tuple
    (size, size_type, direction). Returning size=np.nan signals NoOrder.

    Position sizing:
      - Fixed notional per leg = $10,000 (we use asset A as reference and scale B by hedge_ratio,
        i.e., base pair size = notional / price_A; Asset A units = base, Asset B units = hedge_ratio * base).
      - We place volume-based orders (size_type=0) with explicit number of shares so the
        simulator executes them in a single order rather than many small partial fills.

    Signals and risk management:
      - Entry: on crossing the entry threshold (reduces repeated triggers)
      - Exit: zscore crosses exit_threshold (typically 0.0) -> close positions
      - Stop-loss: |zscore| > stop_threshold -> close positions; do not re-enter while |z| > stop_threshold
      - Cooldown: after any trade on an asset, wait _COOLDOWN_BARS before re-entering
      - Global cap: do not emit more than _MAX_ORDERS orders in total

    Direction encoding (vectorbt enum values):
      - 1 => BUY / LONG
      - 2 => SELL / SHORT

    Args:
        c: Order/Context object. In flexible mode wrapper provides attributes: i (index), col (0 or 1), position_now, cash_now
        close_a, close_b: 1D numpy arrays of closes
        zscore: 1D numpy array of z-score
        hedge_ratio: 1D numpy array of hedge ratio
        entry_threshold: Entry z-score threshold (e.g. 2.0)
        exit_threshold: Exit threshold (e.g. 0.0)
        stop_threshold: Stop-loss threshold (e.g. 3.0)

    Returns:
        Tuple (size, size_type, direction). Use np.nan for size to indicate no order.
    """
    global _LAST_TRADE_IDX, _ORDERS_EMITTED

    # Safeguard: do not emit more than a global cap of orders
    if _ORDERS_EMITTED >= _MAX_ORDERS:
        return (np.nan, 0, 0)

    # Get current index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Current values
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) and not np.isnan(hedge_ratio[i]) else np.nan

    # Current position for this column (number of units/shares). Default to 0.0
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Price for this asset
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Fixed notional per leg (use A as reference to scale pair)
    notional = 10_000.0

    # Compute base number of A shares for one scaled pair such that A notional ~= $10k
    base_shares = notional / price_a if price_a > 0 else 0.0
    b_shares = hr * base_shares if np.isfinite(hr) else 0.0

    # previous z for crossing detections
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Cooldown check: do not re-enter if recently traded this asset
    if (i - _LAST_TRADE_IDX[col]) < _COOLDOWN_BARS:
        # Still allow exit orders to close an existing position
        if pos_now == 0.0:
            return (np.nan, 0, 0)

    # --- Exit / Stop-loss rules (these take priority) ---
    # Stop-loss: close if |z| > stop_threshold
    if pos_now != 0.0 and abs(z) > float(stop_threshold):
        # Close by placing a volume order equal to the current position magnitude
        size = abs(pos_now)
        size_type = 0  # volume-based
        direction = 2 if pos_now > 0 else 1  # sell if long, buy if short
        # record last trade index for this column and the other (pair)
        _LAST_TRADE_IDX[col] = i
        _LAST_TRADE_IDX[1 - col] = i
        _ORDERS_EMITTED += 1
        return (float(size), int(size_type), int(direction))

    # Exit when z-score crosses exit_threshold (commonly 0.0)
    if pos_now != 0.0 and not np.isnan(prev_z):
        crossed_up = (prev_z < exit_threshold) and (z >= exit_threshold)
        crossed_down = (prev_z > exit_threshold) and (z <= exit_threshold)
        if crossed_up or crossed_down:
            size = abs(pos_now)
            size_type = 0  # volume-based
            direction = 2 if pos_now > 0 else 1
            _LAST_TRADE_IDX[col] = i
            _LAST_TRADE_IDX[1 - col] = i
            _ORDERS_EMITTED += 1
            return (float(size), int(size_type), int(direction))

    # --- Entry rules ---
    # Do not enter if currently beyond stop threshold (prevents immediate re-entry after stop-loss)
    if abs(z) > float(stop_threshold):
        return (np.nan, 0, 0)

    # Only open a new position if there is no current position for this asset
    if pos_now == 0.0 and not np.isnan(prev_z):
        # Upper entry: cross above entry_threshold
        crossed_entry_up = (prev_z < entry_threshold) and (z > entry_threshold)
        # Lower entry: cross below -entry_threshold
        crossed_entry_down = (prev_z > -entry_threshold) and (z < -entry_threshold)

        if crossed_entry_up:
            if col == 0:
                # Short asset A by base_shares
                size = base_shares
                if size <= 0 or np.isnan(size):
                    return (np.nan, 0, 0)
                # record trade index for both assets
                _LAST_TRADE_IDX[col] = i
                _LAST_TRADE_IDX[1 - col] = i
                _ORDERS_EMITTED += 1
                return (float(size), 0, 2)
            else:
                # Long asset B by b_shares
                size = b_shares
                if size <= 0 or np.isnan(size):
                    return (np.nan, 0, 0)
                _LAST_TRADE_IDX[col] = i
                _LAST_TRADE_IDX[1 - col] = i
                _ORDERS_EMITTED += 1
                return (float(size), 0, 1)

        if crossed_entry_down:
            if col == 0:
                # Long asset A by base_shares
                size = base_shares
                if size <= 0 or np.isnan(size):
                    return (np.nan, 0, 0)
                _LAST_TRADE_IDX[col] = i
                _LAST_TRADE_IDX[1 - col] = i
                _ORDERS_EMITTED += 1
                return (float(size), 0, 1)
            else:
                # Short asset B by b_shares
                size = b_shares
                if size <= 0 or np.isnan(size):
                    return (np.nan, 0, 0)
                _LAST_TRADE_IDX[col] = i
                _LAST_TRADE_IDX[1 - col] = i
                _ORDERS_EMITTED += 1
                return (float(size), 0, 2)

    # Default: no order
    return (np.nan, 0, 0)
