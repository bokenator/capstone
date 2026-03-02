# Complete implementation for pairs trading strategy using vectorbt

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two price series.

    Args:
        close_a: Price series for asset A (1D array-like).
        close_b: Price series for asset B (1D array-like).
        hedge_lookback: Lookback window (in bars) for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of spread to compute z-score.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of same length as inputs with rolling hedge ratio (slope).
            - 'zscore': np.ndarray of same length with rolling z-score of the spread.

    Notes:
        - Uses only past and current data at each index (no lookahead).
        - For indices with insufficient data, outputs are np.nan.
    """
    # Convert inputs to 1D numpy arrays
    a = np.asarray(close_a, dtype=float).ravel()
    b = np.asarray(close_b, dtype=float).ravel()

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (with intercept) to estimate hedge ratio (slope)
    # For each time t, regress a[t-window+1:t+1] on b[t-window+1:t+1]
    # Use expanding window up to hedge_lookback to avoid NaNs early on
    for t in range(n):
        # Use as many past points as available up to hedge_lookback
        window = min(hedge_lookback, t + 1)
        if window < 2:
            # Need at least 2 points to estimate slope
            continue
        start = t + 1 - window
        y = a[start : t + 1]
        x = b[start : t + 1]

        # If any NaNs in window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        # Use scipy.stats.linregress which fits y = intercept + slope * x
        try:
            res = stats.linregress(x, y)
            slope = float(res.slope)
            hedge_ratio[t] = slope
        except Exception:
            # Fallback: use least squares
            try:
                X = np.vstack([x, np.ones_like(x)]).T
                beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
                hedge_ratio[t] = float(beta)
            except Exception:
                hedge_ratio[t] = np.nan

    # Compute spread using hedge_ratio at each time (no lookahead)
    spread = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(hedge_ratio)

    spread[valid] = a[valid] - hedge_ratio[valid] * b[valid]

    # Rolling mean and std of spread for z-score
    # Use pandas rolling which is causal (uses past and current values)
    spread_s = pd.Series(spread)
    rolling_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    zscore = (spread_s - rolling_mean) / rolling_std

    # Clean up infinities
    zscore_vals = zscore.replace([np.inf, -np.inf], np.nan).values

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore_vals,
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
    """
    Order function for vectorbt flexible multi-asset mode.

    This function implements a pairs trading logic where desired target positions
    (in units) are computed for both assets based on the z-score of the spread and
    the rolling hedge ratio. Orders are returned as (size, size_type, direction)
    tuples where:
        - size_type == 3 => TargetAmount (set target units directly)
        - direction == 0 => LongOnly, 1 => ShortOnly, 2 => Both

    Args:
        c: OrderContext-like object with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: 1D array of asset A close prices
        close_b: 1D array of asset B close prices
        zscore: 1D array of z-score values
        hedge_ratio: 1D array of hedge ratios
        entry_threshold: Threshold to enter trades (default 2.0)
        exit_threshold: Threshold to consider exit (used for zero-crossing detection)
        stop_threshold: Stop-loss threshold (default 3.0)
        notional_per_leg: Base notional per leg (used to compute units for asset A)

    Returns:
        Tuple(size, size_type, direction) or (np.nan, 0, 0) to indicate no order.

    Notes:
        - Uses only current and past information (zscore[i], hedge_ratio[i]).
        - Does not use numba. Designed for use with flexible=True in vectorbt.
    """
    # Defensive attribute reading
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Validate lengths and index
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    h = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If indicators not ready, do nothing
    if np.isnan(z) or np.isnan(h) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target units. We size base units for asset A by notional_per_leg / price_a
    units_a = notional_per_leg / price_a
    units_b = h * units_a

    # Determine if we should exit due to stop loss
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry conditions
        if z > entry_threshold:
            # Short A, Long B
            target_a = -units_a
            target_b = +units_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +units_a
            target_b = -units_b
        else:
            # Check zero crossing for exit: sign change between previous z and current z
            target_a = None
            target_b = None
            if i > 0 and not np.isnan(zscore[i - 1]):
                prev_z = float(zscore[i - 1])
                if prev_z * z < 0:
                    target_a = 0.0
                    target_b = 0.0

    # If no explicit target computed, do nothing
    if target_a is None or target_b is None:
        return (np.nan, 0, 0)

    # Choose target for current column
    desired = target_a if col == 0 else target_b

    # Current position in units (may be 0.0)
    position_now = float(getattr(c, "position_now", 0.0))

    # If already at desired target (within tolerance), do nothing
    if np.isclose(position_now, desired, atol=1e-8, rtol=0.0):
        return (np.nan, 0, 0)

    # Prepare order: use TargetAmount (units) to directly set target units
    size_type = 3  # TargetAmount

    # For target zero, set direction to Both so it can close regardless of side
    if desired == 0.0:
        return (0.0, size_type, 2)  # Both

    # For non-zero desired, return absolute size and direction depending on sign
    if desired > 0:
        return (float(abs(desired)), size_type, 0)  # LongOnly
    else:
        return (float(abs(desired)), size_type, 1)  # ShortOnly
