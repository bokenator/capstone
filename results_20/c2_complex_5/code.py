"""
Pairs trading statistical arbitrage strategy utilities.

Exports:
- compute_spread_indicators: compute rolling hedge ratio and z-score of spread
- order_func: flexible order function for vectorbt (non-numba)

Notes:
- Uses rolling OLS (scipy.stats.linregress) to compute hedge ratio with a given lookback.
- Z-score computed as (spread - rolling_mean)/rolling_std with given lookback.
- Entry/exit logic follows the prompt. Position sizing uses a fixed notional per leg
  to size Asset A, and scales Asset B by the hedge ratio to hedge Asset A.

This module is pure Python (no numba) and returns simple Python tuples expected
by the provided backtest runner wrapper.
"""

from typing import Dict, Tuple

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
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    Args:
        close_a: 1d array of close prices for asset A
        close_b: 1d array of close prices for asset B
        hedge_lookback: lookback window (in bars) for rolling OLS to compute hedge ratio
        zscore_lookback: lookback window (in bars) for rolling mean/std to compute z-score

    Returns:
        dict with keys:
            - 'hedge_ratio': 1d np.ndarray of hedge ratios aligned with input (nan during warmup)
            - 'zscore': 1d np.ndarray of z-score of the spread (nan during warmup)

    Notes:
        - Hedge ratio at index i is computed by regressing A ~ B on the window
          [i - hedge_lookback + 1, ..., i]. If there are fewer than 2 valid
          observations in the window, hedge_ratio[i] = np.nan.
        - Z-score uses a rolling mean and rolling std (population std, ddof=0)
          computed over zscore_lookback periods and requires full window.
    """
    # Basic validation
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1-dimensional arrays")
    if len(close_a) != len(close_b):
        raise ValueError("close_a and close_b must have the same length")
    n = len(close_a)

    # Prepare hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2 to perform regression")

    # Rolling OLS: slope of regression A ~ B
    # For each index i where we have a full window, compute slope using scipy.linregress
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        x = close_b[start_idx : end_idx + 1]
        y = close_a[start_idx : end_idx + 1]

        # Mask invalid values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data points to regress
            continue

        try:
            res = linregress(x[mask], y[mask])
            slope = float(res.slope)
            # If slope is nan or infinite, skip
            if not np.isfinite(slope):
                continue
            hedge_ratio[end_idx] = slope
        except Exception:
            # In case regression fails for numerical reasons, leave as nan
            continue

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Compute rolling mean and std for z-score using pandas for convenience
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to be explicit
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Compute z-score safely
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore_series = (spread_series - roll_mean) / roll_std

    # Replace infinite values with nan
    zscore = np.asarray(zscore_series.fillna(np.nan), dtype=float)

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Flexible order function (non-numba) for pairs trading strategy.

    Expected to be used with the provided flexible wrapper so that it's called
    once per column per bar. Returns a tuple (size, size_type, direction):
        - size: float amount (units or value depending on size_type) or np.nan for NoOrder
        - size_type: int corresponding to vectorbt SizeType enum (Amount=0, Value=1, ...)
        - direction: int corresponding to Direction enum (LongOnly=0, ShortOnly=1, Both=2)

    Logic (per prompt):
        - Entry:
            * zscore > entry_threshold: Short A, Long B (hedge_ratio scaled)
            * zscore < -entry_threshold: Long A, Short B
        - Exit:
            * zscore crosses 0.0: close both positions
            * |zscore| > stop_threshold: stop-loss, close both positions
        - Position sizing:
            * Use notional_per_leg to compute units for Asset A: units_A = notional / price_A
            * Asset B units = hedge_ratio * units_A (to hedge 1 unit of A scaled by units_A)
            * Use SizeType.Amount (0) to send orders in units (shares)

    Notes:
        - The function must be robust to NaNs and warmup periods. If there's
          insufficient data (nan zscore or nan hedge_ratio), it returns NoOrder.
        - c is an order context or a simulated context provided by the wrapper
          and should expose attributes: i (index), col (which asset column),
          position_now (current position in units for that asset) and cash_now.

    Returns:
        (size, size_type, direction) where size=np.nan indicates NoOrder.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Default no-order return
    NO_ORDER: Tuple[float, int, int] = (float("nan"), 0, 0)

    # Defensive checks for index bounds
    if i < 0 or i >= len(zscore):
        return NO_ORDER

    price_a = float(close_a[i]) if (i < len(close_a) and np.isfinite(close_a[i])) else np.nan
    price_b = float(close_b[i]) if (i < len(close_b) and np.isfinite(close_b[i])) else np.nan
    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # Current position for this column (in units). The wrapper sets position_now
    position_now = getattr(c, "position_now", None)
    if position_now is None:
        # Try alternative attribute names that vectorbt might provide
        position_now = getattr(c, "position", 0.0)
    try:
        position_now = float(position_now)
    except Exception:
        position_now = 0.0

    # No action if we don't have valid zscore (warmup)
    if not np.isfinite(z):
        return NO_ORDER

    # Helper to close current position for this column
    def close_position() -> Tuple[float, int, int]:
        # If no position, nothing to do
        if not np.isfinite(position_now) or abs(position_now) < 1e-12:
            return NO_ORDER
        # To close, issue an order in "Both" direction so that the order can reduce
        # current position regardless of its sign.
        size = abs(position_now)
        size_type = 0  # Amount (number of units)
        direction = 2  # Both: allow closing existing position
        return (float(size), int(size_type), int(direction))

    # Stop-loss: if |zscore| > stop_threshold -> close positions
    if np.isfinite(stop_threshold) and abs(z) > float(stop_threshold):
        return close_position()

    # Exit: z-score crosses zero -> close positions
    prev_z = None
    if i - 1 >= 0:
        prev_z = zscore[i - 1]
    if prev_z is not None and np.isfinite(prev_z):
        # Check sign change (cross zero)
        if prev_z * z < 0:
            return close_position()

    # ENTRY logic: only when there is no existing position in this column
    if abs(position_now) < 1e-12:
        # Need valid prices
        if not (np.isfinite(price_a) and np.isfinite(price_b)):
            return NO_ORDER

        # Prevent entries if hedge_ratio is not available
        if not np.isfinite(hr):
            return NO_ORDER

        # Units of asset A based on fixed notional per leg
        if not np.isfinite(notional_per_leg) or notional_per_leg <= 0:
            return NO_ORDER

        # Avoid division by zero
        if price_a == 0 or price_b == 0:
            return NO_ORDER

        units_a = float(notional_per_leg) / price_a
        units_b = float(hr) * units_a

        # If zscore indicates we should SHORT A and LONG B
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (float(units_a), 0, 1)  # Amount, ShortOnly
            else:
                # Long Asset B
                return (float(units_b), 0, 0)  # Amount, LongOnly

        # If zscore indicates we should LONG A and SHORT B
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (float(units_a), 0, 0)  # Amount, LongOnly
            else:
                # Short Asset B
                return (float(units_b), 0, 1)  # Amount, ShortOnly

    # Otherwise, no order
    return NO_ORDER