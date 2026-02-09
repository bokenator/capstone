"""
Pairs trading strategy utilities for vectorbt backtest.

Exports:
- compute_spread_indicators: compute rolling hedge ratio (OLS), spread and z-score
- order_func: flexible order function used by vectorbt.Portfolio.from_order_func (flexible=True)

Notes:
- No numba is used.
- Does not import vbt.portfolio.enums as required.
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
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Price series for asset A (1D numpy array)
        close_b: Price series for asset B (1D numpy array)
        hedge_lookback: Lookback window (in periods) for rolling OLS to estimate hedge ratio
        zscore_lookback: Lookback window for spread rolling mean/std to compute z-score

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of same length with rolling hedge ratio (slope)
            - "spread": np.ndarray of same length with spread = A - hedge_ratio * B
            - "zscore": np.ndarray of same length with z-score of spread

    Notes:
        - Windows with insufficient data will contain np.nan values.
        - Uses scipy.stats.linregress for OLS (A ~ alpha + beta * B) and returns beta as hedge ratio.
    """
    # Input validation and conversion
    if close_a is None or close_b is None:
        raise ValueError("close_a and close_b must be provided")

    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each window ending at i compute slope of regression of A on B
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        end = i + 1
        window_b = b[start:end]
        window_a = a[start:end]

        # Skip window if it contains NaNs
        if np.isnan(window_a).any() or np.isnan(window_b).any():
            hedge_ratio[i] = np.nan
            continue

        # Check variance; if nearly constant, skip
        if np.all(window_b == window_b[0]):
            hedge_ratio[i] = np.nan
            continue

        try:
            res = linregress(window_b, window_a)
            hedge_ratio[i] = float(res.slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = a[valid_idx] - hedge_ratio[valid_idx] * b[valid_idx]

    # Compute rolling mean and std for z-score
    # Use pandas rolling to leverage vectorized operations; require full window (min_periods=zscore_lookback)
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to avoid NaN for small windows when appropriate
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    roll_mean_arr = roll_mean.values
    roll_std_arr = roll_std.values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(roll_mean_arr)) & (~np.isnan(roll_std_arr)) & (roll_std_arr > 0)
    zscore[valid_z] = (spread[valid_z] - roll_mean_arr[valid_z]) / roll_std_arr[valid_z]

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
    Order function for flexible multi-asset vectorbt backtest (pairs trading).

    Expected behavior per bar (called once per asset by the provided wrapper):
    - Entry:
        * If zscore > entry_threshold: short Asset A, long Asset B
        * If zscore < -entry_threshold: long Asset A, short Asset B
      Positions follow the hedge ratio: base units for A are computed from notional_per_leg / price_A,
      and B units are hedge_ratio * base_units_A.

    - Exit:
        * If zscore crosses 0.0 (sign change from previous bar): close both positions
        * If |zscore| > stop_threshold: stop-loss, close both positions

    Position sizing:
    - Fixed notional: notional_per_leg per base unit (base units for A = notional_per_leg / price_A)
      B units are scaled by hedge_ratio.

    Returns:
        Tuple (size, size_type, direction):
            - size: absolute number of units to trade (float). Return np.nan for no order.
            - size_type: integer code for size type. We return 0 which corresponds to absolute size (units).
            - direction: integer code for direction. Use 1 for buy/long and 2 for sell/short (compatible with vectorbt enums).

    Notes:
    - The surrounding wrapper will convert this tuple into vbt order objects.
    - Do NOT return numba objects. Return plain Python types.
    """
    # Basic safety checks
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Bounds check
    if i < 0:
        return (np.nan, 0, 0)

    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    # Read current indicators and prices
    zs = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    pa = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    pb = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # Current position (units) for this asset
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # If any essential input is nan, do not place an order
    if np.isnan(zs) or np.isnan(hr) or np.isnan(pa) or np.isnan(pb):
        return (np.nan, 0, 0)

    # Compute previous z-score for crossing detection
    prev_zs = np.nan
    if i > 0:
        prev_val = zscore[i - 1]
        prev_zs = float(prev_val) if not np.isnan(prev_val) else np.nan

    crossed_zero = False
    if not np.isnan(prev_zs) and not np.isnan(zs) and (prev_zs * zs < 0):
        crossed_zero = True

    # Compute base units: scale of 1 unit of A based on fixed notional per leg
    # This yields unit counts (can be fractional)
    if pa == 0:
        return (np.nan, 0, 0)

    base_a_units = float(notional_per_leg) / pa
    base_b_units = float(hr) * base_a_units

    # Decide target positions for pair
    target_a: float
    target_b: float

    # Stop-loss has priority
    if abs(zs) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    # Crossing zero triggers exit
    elif crossed_zero:
        target_a = 0.0
        target_b = 0.0
    # Entry long/short signals
    elif zs > entry_threshold:
        # Short A, Long B
        target_a = -base_a_units
        target_b = +base_b_units
    elif zs < -entry_threshold:
        # Long A, Short B
        target_a = +base_a_units
        target_b = -base_b_units
    else:
        # No trade signal
        return (np.nan, 0, 0)

    # Select target for this column
    if col == 0:
        target = target_a
        price = pa
    else:
        target = target_b
        price = pb

    # Compute delta in units
    size_delta = target - pos_now

    # If delta is negligible -> no order
    if abs(size_delta) <= 1e-12:
        return (np.nan, 0, 0)

    # Direction: 1 for buy (increase position), 2 for sell (decrease position / short)
    direction = 1 if size_delta > 0 else 2
    size = float(abs(size_delta))

    # Return size (units), size_type=0 (absolute units), direction as integer
    return (size, 0, int(direction))
