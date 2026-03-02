"""
Pairs trading utilities for vectorbt backtests.

Exports:
- compute_spread_indicators
- order_func

Notes:
- No numba used.
- Uses vectorbt enums for size_type and direction integers.
"""
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

import vectorbt as vbt


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Hedge ratio is computed as the slope of regression Price_A ~ Price_B over a
    rolling window of length `hedge_lookback` (ordinary least squares). The
    spread is defined as spread = Price_A - hedge_ratio * Price_B.

    Z-score is computed using rolling mean and std of the spread (window
    `zscore_lookback`). Warmup periods will contain np.nan.

    Args:
        close_a: 1d numpy array of close prices for asset A
        close_b: 1d numpy array of close prices for asset B
        hedge_lookback: lookback for rolling OLS (in bars)
        zscore_lookback: lookback for rolling mean/std of spread

    Returns:
        dict with keys:
            - 'hedge_ratio': numpy array of hedge ratios (same length as inputs)
            - 'spread': numpy array of spreads
            - 'zscore': numpy array of z-scores

    """
    # Basic validation
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1-D arrays")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS slope via covariance/variance formula for numerical stability
    # slope = cov(a, b) / var(b)
    for i in range(hedge_lookback - 1, n):
        win_a = a[i - hedge_lookback + 1 : i + 1]
        win_b = b[i - hedge_lookback + 1 : i + 1]

        # Skip if there are NaNs in the window
        if np.isnan(win_a).any() or np.isnan(win_b).any():
            hedge_ratio[i] = np.nan
            continue

        # Compute covariance and variance
        mean_a = win_a.mean()
        mean_b = win_b.mean()
        cov = ((win_a - mean_a) * (win_b - mean_b)).mean()
        var_b = ((win_b - mean_b) ** 2).mean()

        if var_b == 0 or np.isclose(var_b, 0.0):
            hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = cov / var_b

    # Spread: price_a - hedge_ratio * price_b
    spread = a - hedge_ratio * b

    # Compute rolling mean/std of spread using pandas for convenience
    spread_s = pd.Series(spread)

    rolling_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Z-score
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (spread_s - rolling_mean) / rolling_std

    # Convert to numpy arrays
    result = {
        "hedge_ratio": hedge_ratio,
        "spread": spread_s.values,
        "zscore": zscore.values,
    }

    return result


def _safe_get_position(c: Any, col: int) -> float:
    """Helper to retrieve current position for given column from OrderContext-like object.

    Tries common attribute names used by vectorbt flex context wrapper and
    OrderContext. Falls back to 0.0 if not found.
    """
    # Preferred attribute set by the provided wrapper
    if hasattr(c, "position_now"):
        try:
            return float(getattr(c, "position_now"))
        except Exception:
            # If position_now is array-like (unlikely here), try indexing
            try:
                return float(getattr(c, "position_now")[col])
            except Exception:
                pass

    # Some contexts provide last_position (array)
    if hasattr(c, "last_position"):
        try:
            lp = getattr(c, "last_position")
            return float(lp[col])
        except Exception:
            pass

    # Some contexts provide position (array)
    if hasattr(c, "position"):
        try:
            p = getattr(c, "position")
            return float(p[col])
        except Exception:
            pass

    # As a last resort, try `value_now` or `cash_now` won't give position, so return 0.
    return 0.0


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
    Order function compatible with the flexible wrapper used in the backtest.

    User signature (expected by the runner):
        order_func(c, close_a, close_b, zscore, hedge_ratio,
                   entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

    Returns:
        (size, size_type, direction) tuple. If no order should be placed, return
        (np.nan, 0, 0) where the first element is NaN.

    Behavior (per specification):
    - Entry when zscore > entry_threshold: SHORT Asset A, LONG Asset B
    - Entry when zscore < -entry_threshold: LONG Asset A, SHORT Asset B
    - Exit when zscore crosses 0.0 (sign change) OR |zscore| > stop_threshold
    - Fixed notional sizing: use Value-based orders of `notional_per_leg` per leg
    - No volatility adjustment

    Notes:
    - This function is written to be robust to different context objects (flex
      wrapper or real vectorbt OrderContext).
    """
    # Read index and column
    i = getattr(c, "i", None)
    col = getattr(c, "col", 0)

    if i is None:
        # If context doesn't provide a bar index, we cannot proceed
        return (np.nan, 0, 0)

    # Safety checks on arrays
    if i < 0 or i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If indicators are not ready, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Previous zscore for crossover detection
    prev_z = float(zscore[i - 1]) if i > 0 else np.nan

    # Determine exit conditions
    exit_signal = False
    # Stop-loss
    if np.abs(z) > stop_threshold:
        exit_signal = True
    # Z-score crosses zero
    elif not np.isnan(prev_z) and (prev_z > 0 and z <= 0 or prev_z < 0 and z >= 0):
        exit_signal = True

    pos_now = _safe_get_position(c, int(col))

    # Small tolerance to consider position as zero
    EPS = 1e-9

    # Enums for size type and direction
    SizeType = vbt.Portfolio.SizeType
    Direction = vbt.Portfolio.Direction

    # Exit (close both legs) -> issue target orders to bring position to zero
    if exit_signal:
        if np.abs(pos_now) > EPS:
            # Use TargetAmount to target 0 units (close position)
            size = 0.0
            size_type = int(SizeType.TargetAmount)
            direction = int(Direction.Both)
            return (float(size), int(size_type), int(direction))
        else:
            return (np.nan, 0, 0)

    # Entry signals
    long_a_short_b = z < -entry_threshold
    short_a_long_b = z > entry_threshold

    # If no entry signal, do nothing
    if not (long_a_short_b or short_a_long_b):
        return (np.nan, 0, 0)

    # Only open if we don't already have a position in this asset
    if np.abs(pos_now) > EPS:
        # Already have position in this asset; do not re-enter
        return (np.nan, 0, 0)

    # Use fixed notional per leg (Value size type)
    size = float(notional_per_leg)
    size_type = int(SizeType.Value)

    # Determine direction based on column and signal
    if short_a_long_b:
        # Asset A (col 0) SHORT, Asset B (col 1) LONG
        if int(col) == 0:
            direction = int(Direction.ShortOnly)
        else:
            direction = int(Direction.LongOnly)
    else:
        # long_a_short_b
        if int(col) == 0:
            direction = int(Direction.LongOnly)
        else:
            direction = int(Direction.ShortOnly)

    return (float(size), int(size_type), int(direction))
