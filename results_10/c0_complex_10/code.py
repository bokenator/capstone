"""
Pairs trading indicator and order function implementation.

Provides:
- compute_spread_indicators: computes rolling OLS hedge ratio, spread, and z-score
- order_func: returns order tuples for vectorbt.from_order_func (flexible mode)

Notes:
- No numba is used.
- Does not import vbt enums or nb functions.

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
    """Compute rolling hedge ratio (OLS), spread and z-score for a pair.

    Args:
        close_a: Price array for asset A.
        close_b: Price array for asset B.
        hedge_lookback: Lookback window for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": np.ndarray of hedge ratios (same length as inputs)
            - "spread": np.ndarray of spread values
            - "zscore": np.ndarray of z-score values

    The hedge ratio at time t is estimated by regressing A ~ beta * B + intercept
    over the previous `hedge_lookback` observations (including t). If insufficient
    data or NaNs exist in the window, the hedge ratio is set to NaN.

    Z-score is computed using rolling mean and std of the spread over
    `zscore_lookback` periods (requires full window to produce values).
    """
    # Convert to numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (slope) using covariance formula to avoid heavy dependencies
    # For each window ending at i, compute slope = cov(x, y) / var(x)
    # where x = close_b, y = close_a
    if hedge_lookback <= 1:
        raise ValueError("hedge_lookback must be > 1")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        x = close_b[start : i + 1]
        y = close_a[start : i + 1]

        # Skip if NaNs present in the window
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom <= 0:
            # Can't compute slope (zero variance)
            continue

        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        hedge_ratio[i] = slope

    # Compute spread: A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std of spread using pandas for robust rolling handling
    spread_sr = pd.Series(spread)
    roll_mean = spread_sr.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_sr.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Z-score: (spread - mean) / std
    zscore = (spread_sr - roll_mean) / roll_std

    # Convert back to numpy arrays
    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread_sr.values,
        "zscore": zscore.values,
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
    """Order function for a pairs trading strategy (flexible multi-asset mode).

    This function follows the convention expected by the test harness's
    flexible order wrapper. It is called once per column (asset) per bar.

    Returns a tuple (size, size_type, direction). If no order is to be placed
    for the given column on this bar, return (np.nan, 0, 0).

    Size semantics used here:
    - size_type = 0 => absolute number of units (order size expressed in units)
    - direction = 1 => buy / long
    - direction = 2 => sell / short

    The function computes desired target unit positions for both assets based on
    the z-score and hedge ratio, then issues an order for the current column
    equal to the delta between the desired position and the current position.

    Rules implemented:
    - Entry: zscore > entry_threshold => short A, long B (B units = hedge_ratio * A_units)
             zscore < -entry_threshold => long A, short B
    - Exit: zscore crosses zero => close both legs
    - Stop-loss: |zscore| > stop_threshold => close both legs immediately
    - Position sizing: base A units = notional_per_leg / price_A
                      B units = hedge_ratio * base_A_units (if hedge_ratio available),
                      otherwise fallback to notional_per_leg / price_B

    Args:
        c: Context object with attributes .i (index) and .col (column index) and
           .position_now (current position in units for this column). Flexible
           wrapper provides _FlexOrderContext with these attributes.
        close_a: Price array for asset A.
        close_b: Price array for asset B.
        zscore: Array of z-score values (same length as close arrays).
        hedge_ratio: Array of hedge ratios (same length as close arrays).
        entry_threshold: Entry z-score threshold (positive number).
        exit_threshold: Exit threshold (unused directly; we close on crossing zero)
        stop_threshold: Stop-loss z-score magnitude.
        notional_per_leg: Fixed dollar amount per leg.

    Returns:
        Tuple (size, size_type, direction). If no order, size is np.nan.
    """
    # Constants for size type and directions (integers expected by the wrapper)
    SIZE_TYPE_ABSOLUTE = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safeguard indices
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z_i = zscore[i]
    hr_i = hedge_ratio[i] if (hedge_ratio is not None and len(hedge_ratio) > i) else np.nan

    price_a = float(close_a[i]) if len(close_a) > i else np.nan
    price_b = float(close_b[i]) if len(close_b) > i else np.nan

    # If prices or zscore are NaN, do nothing
    if np.isnan(price_a) or np.isnan(price_b) or np.isnan(z_i):
        return (np.nan, 0, 0)

    # Compute base units for asset A (units corresponding to notional per leg)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    base_units_a = notional_per_leg / price_a

    # Compute desired positions (in units) for both assets
    # Default: no action
    desired_a: float | None = None
    desired_b: float | None = None

    # Priority: stop-loss
    if abs(z_i) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Detect crossing zero compared to previous valid zscore
        crossed_zero = False
        if i > 0:
            prev_z = zscore[i - 1]
            if not np.isnan(prev_z) and prev_z != 0 and not np.isnan(z_i) and z_i != 0:
                if (prev_z < 0 and z_i > 0) or (prev_z > 0 and z_i < 0):
                    crossed_zero = True
        if crossed_zero:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if z_i > entry_threshold:
                # Short A, Long B
                desired_a = -base_units_a
                if not np.isnan(hr_i) and not np.isclose(hr_i, 0.0):
                    # Use hedge ratio to size B units relative to A units
                    desired_b = hr_i * base_units_a
                else:
                    # Fallback: use notional per leg for B
                    desired_b = notional_per_leg / price_b
            elif z_i < -entry_threshold:
                # Long A, Short B
                desired_a = +base_units_a
                if not np.isnan(hr_i) and not np.isclose(hr_i, 0.0):
                    desired_b = -hr_i * base_units_a
                else:
                    desired_b = -(notional_per_leg / price_b)
            else:
                # No entry/exit signal; do nothing
                return (np.nan, 0, 0)

    # Select desired position for current column
    desired_pos = desired_a if col == 0 else desired_b

    # Current position (units) - flexible wrapper provides .position_now
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Compute delta
    delta = desired_pos - pos_now

    # If delta is effectively zero, no order
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    size = float(abs(delta))
    direction = DIRECTION_LONG if delta > 0 else DIRECTION_SHORT

    return (size, int(SIZE_TYPE_ABSOLUTE), int(direction))
