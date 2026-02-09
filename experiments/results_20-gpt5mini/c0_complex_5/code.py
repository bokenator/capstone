"""
Pairs trading strategy implementation for vectorbt backtest.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> Dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> Tuple[float, int, int]

Notes:
- No numba is used.
- Uses rolling OLS (scipy.stats.linregress) to compute hedge ratio.
- Z-score computed from spread using rolling mean/std.
- Entry/exit/stop logic implemented in order_func for flexible multi-asset mode.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio, spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (array-like).
        close_b: Prices for asset B (array-like).
        hedge_lookback: Lookback window for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of hedge ratios (length = len(close_a)).
            - "spread": np.ndarray of spread values (A - hedge_ratio * B).
            - "zscore": np.ndarray of z-score of spread.
            - "spread_mean": rolling mean of spread.
            - "spread_std": rolling std of spread.

    Notes:
        - If insufficient data for a window or NaNs are present, outputs contain NaNs
          for those timestamps.
    """
    # Convert inputs to numpy arrays
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]

    if n == 0:
        return {
            "hedge_ratio": np.array([], dtype=float),
            "spread": np.array([], dtype=float),
            "zscore": np.array([], dtype=float),
            "spread_mean": np.array([], dtype=float),
            "spread_std": np.array([], dtype=float),
        }

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (regress A on B) to obtain hedge ratio (slope)
    # Use linregress on each rolling window; skip windows with insufficient non-NaN points
    for t in range(hedge_lookback - 1, n):
        start = t - hedge_lookback + 1
        x = close_b[start : t + 1]
        y = close_a[start : t + 1]
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) >= 2:
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
                hedge_ratio[t] = float(slope)
            except Exception:
                hedge_ratio[t] = np.nan
        else:
            hedge_ratio[t] = np.nan

    # Compute spread
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score. Use pandas rolling which handles NaNs.
    s = pd.Series(spread)
    spread_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to be consistent/stable
    spread_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
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
    """Order function for flexible multi-asset pairs trading.

    This function is called by the flexible wrapper for each column (asset) at each bar.

    Args:
        c: Order context (has attributes: i (bar index), col (column index), position_now, cash_now/value_now)
        close_a: ndarray of close prices for asset A
        close_b: ndarray of close prices for asset B
        zscore: ndarray of zscore values
        hedge_ratio: ndarray of hedge ratios
        entry_threshold: entry threshold for z-score (e.g., 2.0)
        exit_threshold: exit threshold (typically 0.0). Here used for crossing 0 logic.
        stop_threshold: stop-loss threshold for absolute z-score
        notional_per_leg: dollar notional per leg (used as base notional for asset A)

    Returns:
        Tuple of (size, size_type, direction):
            - size: float number of units (shares) to buy/sell, or np.nan for no order
            - size_type: int, 0 = absolute size in units
            - direction: int, 1 = buy (long), 2 = sell (short), 0 = no order

    Notes on sizing logic:
        - Base units for asset A are computed as notional_per_leg / price_A.
        - Asset B units = abs(hedge_ratio) * units_A. The sign of hedge_ratio is used to determine
          whether B's directional intent is reversed.
        - Entry signals are generated when zscore crosses thresholds. Closing happens when zscore
          crosses zero or exceeds stop_threshold in absolute value.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely extract values for this bar
    if i < 0:
        return (np.nan, 0, 0)

    # ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()
    zscore = np.asarray(zscore, dtype=float).ravel()
    hedge_ratio = np.asarray(hedge_ratio, dtype=float).ravel()

    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]
    curr_z = zscore[i]
    prev_z = zscore[i - 1] if i > 0 else np.nan
    h = hedge_ratio[i]

    # Current position for this asset (units/shares). If missing, assume 0
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Basic sanity checks on prices and hedge ratio
    if np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute desired units for asset A and B (units_A is base)
    units_a = notional_per_leg / price_a if notional_per_leg is not None and price_a > 0 else 0.0
    # If hedge ratio is NaN, skip issuing an entry for B; however, we still can compute for A
    if np.isnan(h):
        units_b = np.nan
    else:
        units_b = abs(h) * units_a

    # Determine signals
    entry_short_spread = (not np.isnan(curr_z)) and (curr_z > entry_threshold)
    entry_long_spread = (not np.isnan(curr_z)) and (curr_z < -entry_threshold)

    # Crossing zero: prev_z > 0 and curr_z <= 0, or prev_z < 0 and curr_z >= 0
    exit_cross_zero = False
    if not np.isnan(prev_z) and not np.isnan(curr_z):
        if (prev_z > 0 and curr_z <= exit_threshold) or (prev_z < 0 and curr_z >= exit_threshold):
            exit_cross_zero = True
        # Also treat exact equality at current bar as crossing
        if curr_z == exit_threshold and prev_z != exit_threshold:
            exit_cross_zero = True

    stop_loss = (not np.isnan(curr_z)) and (abs(curr_z) > stop_threshold)

    SIZE_TYPE_ABS = 0  # absolute number of units/shares

    # No-op convenience
    NO_ORDER: Tuple[float, int, int] = (np.nan, 0, 0)

    # Column 0: asset A
    if col == 0:
        # Entry: if no existing position
        if pos_now == 0.0:
            if entry_short_spread:
                # Short asset A -> sell
                size = float(units_a) if units_a > 0 else np.nan
                if np.isnan(size) or size == 0:
                    return NO_ORDER
                return (size, SIZE_TYPE_ABS, 2)

            if entry_long_spread:
                # Long asset A -> buy
                size = float(units_a) if units_a > 0 else np.nan
                if np.isnan(size) or size == 0:
                    return NO_ORDER
                return (size, SIZE_TYPE_ABS, 1)

            return NO_ORDER

        # If we have a position, check exit/stop
        if pos_now != 0.0 and (exit_cross_zero or stop_loss):
            # Close entire position
            size = float(abs(pos_now))
            if size == 0.0:
                return NO_ORDER
            # If currently long (>0) then sell to close (2), if short (<0) then buy to close (1)
            direction = 2 if pos_now > 0 else 1
            return (size, SIZE_TYPE_ABS, direction)

        return NO_ORDER

    # Column 1: asset B
    if col == 1:
        # If hedge ratio is NaN or units_b invalid, do nothing
        if np.isnan(units_b) or units_b == 0.0:
            # However, if we have an existing position on B we may want to close when exit/stop
            if pos_now != 0.0 and (exit_cross_zero or stop_loss):
                size = float(abs(pos_now))
                direction = 2 if pos_now > 0 else 1
                return (size, SIZE_TYPE_ABS, direction)
            return NO_ORDER

        # Entry: if no existing position on B
        if pos_now == 0.0:
            if entry_short_spread:
                # For shorting the spread we want B to be long if hedge ratio positive
                if h >= 0:
                    direction = 1
                else:
                    direction = 2
                size = float(units_b)
                return (size, SIZE_TYPE_ABS, direction)

            if entry_long_spread:
                # For longing the spread we want B to be short if hedge ratio positive
                if h >= 0:
                    direction = 2
                else:
                    direction = 1
                size = float(units_b)
                return (size, SIZE_TYPE_ABS, direction)

            return NO_ORDER

        # If we have a position, check exit/stop
        if pos_now != 0.0 and (exit_cross_zero or stop_loss):
            size = float(abs(pos_now))
            direction = 2 if pos_now > 0 else 1
            return (size, SIZE_TYPE_ABS, direction)

        return NO_ORDER

    # Default: no order
    return (np.nan, 0, 0)
