"""
Pairs trading strategy implementation for vectorbt backtesting.

Exports:
- compute_spread_indicators
- order_func

Implements rolling OLS hedge ratio, spread, z-score and an order function
compatible with vectorbt flexible order_func wrapper (no numba).

Author: ChatGPT (adapted for testing harness)
"""
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import linregress
import vectorbt as vbt


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling OLS hedge ratio and z-score of the spread.

    Args:
        close_a: Prices for asset A as 1D numpy array.
        close_b: Prices for asset B as 1D numpy array.
        hedge_lookback: Lookback window for OLS regression (max window).
                        For early bars (< lookback) an expanding window is used.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        A dict containing at least:
            - 'hedge_ratio': numpy array of hedge ratios (same length as inputs)
            - 'zscore': numpy array of z-scores (same length as inputs)
            - 'spread': numpy array of spreads (same length as inputs)

    Notes:
        - Uses only past and present data at each time step (no lookahead).
        - Ensures no NaNs after minimal warmup by filling initial hedge_ratio from
          the earliest computable values.
    """
    # Convert inputs to 1D float numpy arrays
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]

    hedge_ratio = np.empty(n, dtype=float)
    hedge_ratio.fill(np.nan)

    spread = np.empty(n, dtype=float)
    spread.fill(np.nan)

    # Compute rolling OLS slope (hedge ratio) using available data up to time t
    for t in range(n):
        start = 0 if hedge_lookback is None else max(0, t - hedge_lookback + 1)
        x = close_b[start : t + 1]
        y = close_a[start : t + 1]

        # Need at least 2 points to compute slope; otherwise use previous value or 1.0
        if x.size < 2 or np.all(np.isnan(x)) or np.all(np.isnan(y)):
            if t == 0:
                hedge_ratio[t] = 1.0
            else:
                hedge_ratio[t] = hedge_ratio[t - 1]
            continue

        # If x is constant, slope cannot be determined; fall back to previous value
        if np.nanstd(x) == 0:
            hedge_ratio[t] = hedge_ratio[t - 1] if t > 0 else 1.0
            continue

        # Perform OLS regression of y ~ x (y = intercept + slope * x)
        try:
            res = linregress(x, y)
            slope = float(res.slope)
            if np.isnan(slope):
                # fallback
                slope = hedge_ratio[t - 1] if t > 0 else 1.0
        except Exception:
            slope = hedge_ratio[t - 1] if t > 0 else 1.0

        hedge_ratio[t] = slope

    # Compute spread using the hedge_ratio at time t
    spread = close_a - hedge_ratio * close_b

    # Compute rolling mean and std for z-score using expanding windows until lookback
    zscore = np.empty(n, dtype=float)
    zscore.fill(np.nan)

    for t in range(n):
        start = max(0, t - zscore_lookback + 1)
        window = spread[start : t + 1]

        # ignore NaNs in window (shouldn't be many), but require at least 1 value
        if window.size == 0 or np.all(np.isnan(window)):
            zscore[t] = 0.0
            continue

        mean = np.nanmean(window)
        std = np.nanstd(window, ddof=0)

        if std == 0 or np.isnan(std):
            # If no variation, define zscore as 0
            zscore[t] = 0.0
        else:
            zscore[t] = (spread[t] - mean) / std

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
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Flexible order function for pairs trading.

    This function returns a tuple (size, size_type, direction) for the asset
    indicated by c.col (0 = asset_a, 1 = asset_b). It is intended to be used
    with a wrapper that calls it once per column per bar (flexible mode).

    Rules implemented:
    - Entry: |zscore| > entry_threshold
        - zscore > threshold: Short A, Long B
        - zscore < -threshold: Long A, Short B
    - Exit: zscore crosses exit_threshold (0.0) or |zscore| > stop_threshold
    - Position sizing: Notional per leg for asset A = notional_per_leg
      -> size_a = notional_per_leg / price_a
      -> size_b = abs(hedge_ratio) * size_a (keeps B scaled by hedge ratio)

    Returns:
        (size, size_type, direction)
        - size: positive float amount (number of units when size_type is Amount)
        - size_type: vbt.portfolio.enums.SizeType (use Amount)
        - direction: vbt.portfolio.enums.Direction (LongOnly/ShortOnly/Both)

    When no order is needed, returns (np.nan, SizeType.Amount, Direction.Both)
    (size is nan -> interpreted as NoOrder by the wrapper).
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely pull current and previous zscore and hedge ratio
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, vbt.portfolio.enums.SizeType.Amount, vbt.portfolio.enums.Direction.Both)

    curr_z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else 0.0

    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else 1.0

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Compute notional-based size for asset A (units)
    size_a_units = (notional_per_leg / price_a) if price_a > 0 else 0.0
    size_b_units = abs(hr) * size_a_units

    # Determine signals
    is_stop = np.isfinite(curr_z) and (abs(curr_z) > stop_threshold)

    # zscore crosses exit_threshold (usually 0.0)
    crossed_zero = False
    if i > 0 and np.isfinite(curr_z) and np.isfinite(prev_z):
        crossed_zero = (prev_z > exit_threshold and curr_z <= exit_threshold) or (
            prev_z < exit_threshold and curr_z >= exit_threshold
        )

    enter_short_A_long_B = np.isfinite(curr_z) and (curr_z > entry_threshold)
    enter_long_A_short_B = np.isfinite(curr_z) and (curr_z < -entry_threshold)

    # Current position for this column (units). In flexible wrapper we get position_now
    pos_now = float(getattr(c, "position_now", 0.0))

    SizeType = vbt.portfolio.enums.SizeType
    Direction = vbt.portfolio.enums.Direction

    # Helper to close current position for the given column
    def close_position(size_now: float) -> Tuple[float, int, int]:
        if size_now > 0:
            # currently long -> sell to close (use Both to allow closing)
            return (abs(size_now), SizeType.Amount, Direction.Both)
        elif size_now < 0:
            # currently short -> buy to close
            return (abs(size_now), SizeType.Amount, Direction.Both)
        else:
            return (np.nan, SizeType.Amount, Direction.Both)

    # Exit conditions: stop-loss or crossing zero -> close if in position
    if is_stop or crossed_zero:
        return close_position(pos_now)

    # Entry conditions: only enter when not already in position for the column
    if pos_now == 0:
        if enter_short_A_long_B:
            if col == 0:
                # Short Asset A
                return (size_a_units, SizeType.Amount, Direction.ShortOnly)
            else:
                # Long Asset B (scaled by hedge ratio)
                return (size_b_units, SizeType.Amount, Direction.LongOnly)

        if enter_long_A_short_B:
            if col == 0:
                # Long Asset A
                return (size_a_units, SizeType.Amount, Direction.LongOnly)
            else:
                # Short Asset B
                return (size_b_units, SizeType.Amount, Direction.ShortOnly)

    # No action by default
    return (np.nan, vbt.portfolio.enums.SizeType.Amount, vbt.portfolio.enums.Direction.Both)
