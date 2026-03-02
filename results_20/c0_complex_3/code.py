"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- No numba is used.
- Uses rolling OLS (numpy.polyfit) to compute hedge ratio.
- Sizes orders using fixed notional per leg and preserves hedge ratio units when computing the second leg's size.

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
    """Compute rolling hedge ratio and z-score for the spread.

    Args:
        close_a: Prices for asset A as 1D array.
        close_b: Prices for asset B as 1D array.
        hedge_lookback: Lookback window (in bars) for rolling OLS to compute hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dict with keys:
            - 'hedge_ratio': 1D array of rolling hedge ratios (slope of regressing A on B).
            - 'zscore': 1D array of z-scores of the spread.
            - 'spread': 1D array of spread values (Price_A - hedge_ratio * Price_B).
            - 'roll_mean': rolling mean of spread used for zscore.
            - 'roll_std': rolling std of spread used for zscore.

    Notes:
        - For indices where there is insufficient data for the rolling computations,
          values are set to np.nan.
        - Uses numpy.polyfit for OLS regression of A on B on each rolling window.
    """
    # Ensure numpy arrays of float
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    # Initialize hedge ratio and intercept arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)
    intercept = np.full(n, np.nan, dtype=float)

    # Rolling OLS: regress A on B for each window
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        a_win = a[start : i + 1]
        b_win = b[start : i + 1]

        # Skip windows with NaNs
        if np.isnan(a_win).any() or np.isnan(b_win).any():
            hedge_ratio[i] = np.nan
            intercept[i] = np.nan
            continue

        # If b_win has zero variance, slope is undefined -> set to nan
        if np.allclose(b_win, b_win[0]):
            hedge_ratio[i] = np.nan
            intercept[i] = np.nan
            continue

        try:
            # Fit linear regression: a = slope * b + intercept
            slope, inter = np.polyfit(b_win, a_win, 1)
            hedge_ratio[i] = float(slope)
            intercept[i] = float(inter)
        except Exception:
            hedge_ratio[i] = np.nan
            intercept[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = a[valid_hr] - hedge_ratio[valid_hr] * b[valid_hr]

    # Rolling mean and std of spread for z-score
    # Use pandas rolling to handle min_periods and NaNs cleanly
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to avoid division by zero differences
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score where possible
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(roll_mean)) & (~np.isnan(roll_std))
    # Avoid division by zero
    nonzero = valid & (roll_std > 0)
    zscore[nonzero] = (spread[nonzero] - roll_mean[nonzero]) / roll_std[nonzero]
    # If std is zero (constant spread), set zscore to 0
    zero_std_mask = valid & (roll_std == 0)
    zscore[zero_std_mask] = 0.0

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
        "roll_mean": roll_mean,
        "roll_std": roll_std,
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
    """Order function for vectorbt flexible multi-asset pairs trading.

    This function follows the strategy logic described:
    - Entry: zscore > entry_threshold -> Short A, Long B (hedge_ratio units)
             zscore < -entry_threshold -> Long A, Short B (hedge_ratio units)
    - Exit: zscore crosses 0 -> close both legs
    - Stop-loss: |zscore| > stop_threshold -> close both legs

    Position sizing: fixed notional_per_leg for asset A. Asset B size is set
    based on hedge_ratio to maintain unit hedge: n_A = notional_per_leg / price_A,
    n_B = hedge_ratio * n_A (sign determines direction). Order sizes are expressed
    as notional (size_type = 1).

    Returns:
        (size, size_type, direction)
        - size: float (notional if size_type == 1)
        - size_type: int (1 -> Notional). Use 0/np.nan to indicate NoOrder.
        - direction: int (1 -> Long, 2 -> Short). 0 reserved for NoOrder.

    Notes:
        - The function returns (np.nan, 0, 0) when no order should be placed.
        - It uses c.i (current index) and c.col (0 for asset_a, 1 for asset_b).
        - c.position_now is used to determine if a position exists and to compute close sizes.
    """
    i = int(getattr(c, "i", 0))

    # Safety bounds
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Current prices
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current indicators
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not available -> no order
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Previous z (for crossing detection)
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Detect zero crossing
    crossed_zero = False
    if not np.isnan(prev_z):
        if (prev_z < 0 and z >= 0) or (prev_z > 0 and z <= 0):
            crossed_zero = True

    # Current position in this column (units). In flexible wrapper this will be scalar.
    pos_now = float(getattr(c, "position_now", 0.0))
    pos_is_open = not np.isclose(pos_now, 0.0)

    col = int(getattr(c, "col", 0))

    # Helper to return no order
    def no_order() -> Tuple[float, int, int]:
        return (np.nan, 0, 0)

    # If stop-loss or exit condition and position exists -> close
    if pos_is_open and (abs(z) > stop_threshold or crossed_zero):
        # Assume pos_now is stored in units. Convert to notional for the order size.
        price = price_a if col == 0 else price_b
        notional_size = abs(pos_now) * price if not np.isnan(pos_now) else notional_per_leg

        # Determine direction to close: if currently long (pos_now > 0) -> place short order to close (2)
        # if currently short (pos_now < 0) -> place long order to close (1)
        direction = 1 if pos_now < 0 else 2
        # Safety: if notional_size is 0, skip
        if notional_size <= 0:
            return no_order()
        return (float(notional_size), 1, int(direction))

    # If no position exists, consider entries
    if not pos_is_open:
        n_a_units = notional_per_leg / price_a if price_a > 0 else 0.0

        # Entry: z-score triggers
        if z > entry_threshold:
            # Short A, Long B (units: A = -n_a_units, B = +hr * n_a_units)
            if col == 0:
                target_units = -n_a_units
            else:
                target_units = hr * n_a_units
        elif z < -entry_threshold:
            # Long A, Short B (A = +n_a_units, B = -hr * n_a_units)
            if col == 0:
                target_units = n_a_units
            else:
                target_units = -hr * n_a_units
        else:
            return no_order()

        # If target_units == 0, skip
        if np.isclose(target_units, 0.0):
            return no_order()

        # Express order as notional size (size_type = 1)
        price = price_a if col == 0 else price_b
        notional_size = abs(target_units) * price
        direction = 1 if target_units > 0 else 2
        # Sanity checks
        if notional_size <= 0:
            return no_order()

        return (float(notional_size), 1, int(direction))

    # Otherwise: keep existing position (no action)
    return no_order()
