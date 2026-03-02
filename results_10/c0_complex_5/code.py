# -*- coding: utf-8 -*-
"""
Pairs trading helper functions for vectorbt backtests.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (lookback default 60)
- Spread = A - hedge_ratio * B
- Z-score computed with rolling mean/std (lookback default 20)
- Entry: |z| > entry_threshold -> open pair (A vs B)
- Exit: z crosses zero OR |z| > stop_threshold -> close pair
- Position sizing: fixed notional per base leg (notional_per_leg)

CRITICAL: Do NOT use numba in these functions.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score for a pairs spread.

    Args:
        close_a: Prices of asset A (1D numpy array).
        close_b: Prices of asset B (1D numpy array).
        hedge_lookback: Lookback in periods for rolling OLS to compute hedge ratio.
        zscore_lookback: Lookback in periods for rolling mean/std of spread.

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of rolling hedge ratios (same length as inputs)
            - "zscore": np.ndarray of z-score of the spread (same length as inputs)

    Notes:
        - Uses ordinary least squares (linear regression) on each rolling window.
        - Handles NaNs and insufficient data by filling with np.nan for those indices.
    """
    # Validate inputs
    if not isinstance(close_a, np.ndarray):
        close_a = np.asarray(close_a)
    if not isinstance(close_b, np.ndarray):
        close_b = np.asarray(close_b)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    n = len(close_a)
    if len(close_b) != n:
        raise ValueError("close_a and close_b must have the same length")

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: regress A on B in each window to get slope (hedge ratio)
    for i in range(hedge_lookback - 1, n):
        window_slice = slice(i - hedge_lookback + 1, i + 1)
        y = close_a[window_slice]
        x = close_b[window_slice]

        # Handle NaNs in window: require at least half non-na values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if valid_mask.sum() < max(2, hedge_lookback // 2):
            hedge_ratio[i] = np.nan
            continue

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # If x is constant (zero variance), skip
        if np.allclose(x_valid, x_valid[0]):
            hedge_ratio[i] = np.nan
            continue

        # Linear regression (y = slope * x + intercept)
        try:
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    valid_prices = np.isfinite(close_a) & np.isfinite(close_b)
    valid = valid_hr & valid_prices
    spread[valid] = close_a[valid] - hedge_ratio[valid] * close_b[valid]

    # Compute rolling mean and std for z-score using pandas rolling
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    zscore = (spread_series - roll_mean) / roll_std
    zscore_arr = zscore.to_numpy(dtype=float)

    # Where roll_std is zero or NaN, set zscore to NaN
    zscore_arr[~np.isfinite(zscore_arr)] = np.nan

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore_arr,
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
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """Order function for vectorbt flexible multi-asset backtest.

    Signature expected by the backtest wrapper:
        order_func(c, close_a, close_b, zscore, hedge_ratio,
                   entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

    Returns a tuple (size, size_type, direction) where:
        - size: float (for absolute size this is number of shares; for target value this is dollars)
        - size_type: int (we use 0 for absolute number of shares)
        - direction: int (0 used as Any to allow flips)

    The wrapper will treat a returned tuple with size = np.nan as NoOrder.

    Logic implemented:
        - Entry when zscore > entry_threshold: Short A, Long B
        - Entry when zscore < -entry_threshold: Long A, Short B
        - Exit when zscore crosses zero or |zscore| > stop_threshold: close both legs
        - Position sizing: base leg (A) uses notional_per_leg; B is scaled to match hedge ratio

    CRITICAL: Do not use numba or vectorbt nb helpers here.
    """
    # Use absolute number of shares for sizing (size_type = 0)
    SIZE_TYPE_ABSOLUTE = 0
    DIRECTION_ANY = 0

    i = int(getattr(c, "i", 0))

    # Safeguard array bounds
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan
    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # If any essential value is NaN, do nothing
    if not (np.isfinite(price_a) and np.isfinite(price_b) and np.isfinite(z) and np.isfinite(hr)):
        return (np.nan, 0, 0)

    # Previous z-score for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Determine signals
    enter_short_pair = z > entry_threshold
    enter_long_pair = z < -entry_threshold
    stop_loss = abs(z) > stop_threshold

    cross_zero = False
    if np.isfinite(prev_z):
        cross_zero = (prev_z < 0 and z > 0) or (prev_z > 0 and z < 0)
    # Also consider being very close to zero as exit (supports exit_threshold > 0)
    close_to_zero = abs(z) <= exit_threshold

    should_exit = stop_loss or cross_zero or close_to_zero

    # Current position in shares (simulated context provides position_now)
    pos_now = getattr(c, "position_now", 0.0)
    # If position_now is an array (some contexts), take the scalar for this column
    if isinstance(pos_now, (list, tuple, np.ndarray)):
        try:
            pos_now = float(pos_now[c.col]) if hasattr(c, "col") else float(pos_now[0])
        except Exception:
            # Fallback
            pos_now = float(pos_now[0]) if len(pos_now) > 0 else 0.0
    else:
        try:
            pos_now = float(pos_now)
        except Exception:
            pos_now = 0.0

    # Column indicates which asset we are producing order for: 0 -> asset_a, 1 -> asset_b
    col = int(getattr(c, "col", 0))

    # Helper: return NoOrder
    def no_order() -> Tuple[float, int, int]:
        return (np.nan, 0, 0)

    # Tolerance for comparing floats (shares)
    TOL = 1e-8

    # If we should exit (close both legs), issue close order for any non-zero position
    if should_exit:
        if np.isfinite(pos_now) and abs(pos_now) > TOL:
            return (0.0, SIZE_TYPE_ABSOLUTE, DIRECTION_ANY)
        return no_order()

    # Entry logic
    if enter_short_pair or enter_long_pair:
        # Base share size for A (absolute shares, positive magnitude)
        if price_a == 0:
            return no_order()
        base_shares = notional_per_leg / price_a

        if col == 0:
            # Asset A
            desired_shares_a = base_shares if enter_long_pair else -base_shares
            # If already approximately at desired shares, do nothing
            if abs(pos_now - desired_shares_a) <= TOL:
                return no_order()
            return (float(desired_shares_a), SIZE_TYPE_ABSOLUTE, DIRECTION_ANY)

        elif col == 1:
            # Asset B: size = - hedge_ratio * shares_A (so signs are opposite)
            desired_shares_b = -hr * (base_shares if enter_long_pair else -base_shares)
            # Simplify expression: desired_shares_b = -hr * desired_shares_a
            # If already approximately at desired shares, do nothing
            if abs(pos_now - desired_shares_b) <= TOL:
                return no_order()
            return (float(desired_shares_b), SIZE_TYPE_ABSOLUTE, DIRECTION_ANY)

    # Default: no action
    return no_order()
