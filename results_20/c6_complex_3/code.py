# Pairs trading strategy implementation for vectorbt
# Exports: compute_spread_indicators, order_func

from typing import Dict, Any, Tuple

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
        close_a: 1D array of close prices for asset A.
        close_b: 1D array of close prices for asset B.
        hedge_lookback: Lookback window for rolling OLS (uses expanding window when
                        there are fewer points than lookback to avoid NaNs).
        zscore_lookback: Window for rolling mean/std of the spread used to compute z-score.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-scores (same length as inputs)
            - 'spread': np.ndarray of spread values (same length as inputs)

    Notes:
        - Uses only past and current data for rolling computations (no lookahead).
        - Avoids NaNs by falling back to sensible defaults when insufficient variation is present.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    hedge_ratio = np.zeros(n, dtype=float)

    # Rolling OLS (regress A ~ B) using an expanding window until hedge_lookback is reached
    for t in range(n):
        start = 0 if t - hedge_lookback + 1 < 0 else t - hedge_lookback + 1
        y = a[start : t + 1]
        x = b[start : t + 1]

        # Need at least 2 points for regression; otherwise fallback to 0.0
        if x.size >= 2:
            # If x has zero variance, slope is ill-defined -> fallback
            if np.nanstd(x) == 0.0:
                slope = 0.0
            else:
                try:
                    lr = linregress(x, y)
                    slope = float(lr.slope) if np.isfinite(lr.slope) else 0.0
                except Exception:
                    slope = 0.0
        else:
            slope = 0.0

        hedge_ratio[t] = slope

    # Compute spread using current hedge ratio (no lookahead)
    spread = a - hedge_ratio * b

    # Rolling mean and std of spread for z-score (use min_periods=1 to avoid NaNs early)
    s = pd.Series(spread)
    roll_mean = s.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy(dtype=float)
    # Use population std (ddof=0) so single-value windows yield std=0.0 instead of NaN
    roll_std = s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy(dtype=float)

    # Compute z-score safely
    zscore = np.zeros(n, dtype=float)
    # Avoid division by zero
    nonzero_mask = roll_std > 0
    zscore[nonzero_mask] = (spread[nonzero_mask] - roll_mean[nonzero_mask]) / roll_std[nonzero_mask]
    zscore[~nonzero_mask] = 0.0

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
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

    The wrapper provided by the backtest will call this function for each column
    (asset) and collect orders for the current bar. This function returns a tuple
    (size, size_type, direction) or (np.nan, ..., ...) to signal no order.

    Behavior implemented:
    - Entry: when z-score > entry_threshold -> SHORT A, LONG B (scaled by hedge_ratio)
             when z-score < -entry_threshold -> LONG A, SHORT B (scaled by hedge_ratio)
      Position sizing in units (Amount):
         units_a = notional_per_leg / price_a
         units_b = abs(hedge_ratio) * units_a
      Direction of B is flipped if hedge_ratio is negative.
    - Exit: when z-score crosses 0 -> target amount 0 for both legs
    - Stop-loss: when |z-score| > stop_threshold -> target amount 0 for both legs

    Returns:
        (size, size_type, direction)
        - size: float (number of units for SizeType.Amount or target units for TargetAmount)
        - size_type: int (SizeType enum value)
        - direction: int (Direction enum value)

    Notes:
        - Uses only data up to index c.i (no lookahead).
        - Uses integer enum values directly to avoid importing vbt enums here.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safe guards
    if i < 0:
        return (np.nan, 0, 0)

    # Pull current z-score and hedge ratio (use safe defaults if out-of-bounds)
    try:
        z = float(zscore[i])
    except Exception:
        z = 0.0
    try:
        h = float(hedge_ratio[i])
    except Exception:
        h = 0.0

    # Current prices
    pa = float(close_a[i])
    pb = float(close_b[i])

    # Current position for this column (units). If missing, assume 0.
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Enum integer mappings (from vectorbt.portfolio.enums)
    SIZE_TYPE_AMOUNT = 0        # Amount (units)
    SIZE_TYPE_TARGET_AMOUNT = 3 # TargetAmount
    DIRECTION_LONG = 0         # LongOnly
    DIRECTION_SHORT = 1        # ShortOnly
    DIRECTION_BOTH = 2         # Both

    # Compute desired units for asset A and B (units, not value)
    units_a = (notional_per_leg / pa) if pa != 0 else 0.0
    units_b = abs(h) * units_a

    # Tolerance for considering position as zero
    zero_tol = 1e-8

    # STOP-LOSS: if |z| > stop_threshold -> close both legs (target amount = 0)
    if abs(z) > float(stop_threshold):
        if abs(pos_now) > zero_tol:
            return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)
        else:
            return (np.nan, 0, 0)

    # EXIT: if z-score crossed zero (from prev bar to current), close if in position
    crossed_zero = False
    if i > 0:
        try:
            prev_z = float(zscore[i - 1])
            # Crossing zero includes touching zero
            if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
                crossed_zero = True
        except Exception:
            crossed_zero = False

    if crossed_zero:
        if abs(pos_now) > zero_tol:
            return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)
        else:
            return (np.nan, 0, 0)

    # ENTRY: open positions when |z| breaches entry_threshold
    # Only open if currently flat on this leg
    is_flat = abs(pos_now) <= zero_tol

    if z > float(entry_threshold):
        # SHORT A, LONG B (scaling B by hedge ratio magnitude)
        if col == 0:
            # Asset A: short
            if is_flat:
                return (units_a, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: long if hedge ratio positive, else short
            if is_flat:
                if h >= 0:
                    return (units_b, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
                else:
                    return (units_b, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
            else:
                return (np.nan, 0, 0)

    if z < -float(entry_threshold):
        # LONG A, SHORT B
        if col == 0:
            if is_flat:
                return (units_a, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            else:
                return (np.nan, 0, 0)
        else:
            if is_flat:
                # Asset B: short if hedge ratio positive, else long
                if h >= 0:
                    return (units_b, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
                else:
                    return (units_b, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            else:
                return (np.nan, 0, 0)

    # No order by default
    return (np.nan, 0, 0)
