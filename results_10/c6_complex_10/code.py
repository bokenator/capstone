"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> tuple[float,int,int]

Notes:
- No numba used.
- Uses rolling OLS (via numpy.polyfit) with expanding behavior for warmup periods (uses available history if less than lookback).
- Z-score uses rolling mean/std with available history (min periods = 1). Std=0 handled to avoid NaNs.
- Orders use Value sizing for entries (notional per leg) and TargetAmount (0) to close positions.
"""
from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np

# Constants matching vectorbt enums (numeric values verified via docs)
SIZE_TYPE_AMOUNT = 0
SIZE_TYPE_VALUE = 1
SIZE_TYPE_PERCENT = 2
SIZE_TYPE_TARGET_AMOUNT = 3
SIZE_TYPE_TARGET_VALUE = 4
SIZE_TYPE_TARGET_PERCENT = 5

DIRECTION_LONGONLY = 0
DIRECTION_SHORTONLY = 1
DIRECTION_BOTH = 2


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    The implementation is causal (no lookahead): for time t we only use data up to
    and including t. When there is not enough history to fill the requested
    lookback, we use the available history (expanding behaviour). This ensures
    there are no NaNs after warmup if a reasonable warmup is chosen.

    Args:
        close_a: Prices of asset A (array-like).
        close_b: Prices of asset B (array-like).
        hedge_lookback: Lookback (in bars) for rolling OLS hedge ratio.
        zscore_lookback: Lookback (in bars) for z-score rolling mean/std.

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray, same length as inputs
            - "zscore": np.ndarray, same length as inputs
    """
    # Convert inputs to numpy arrays
    ca = np.asarray(close_a, dtype=float)
    cb = np.asarray(close_b, dtype=float)

    if ca.shape != cb.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = ca.shape[0]

    # Pre-allocate arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Fallback initial hedge ratio
    last_hr = 1.0

    for t in range(n):
        # Rolling OLS (regress A on B) using available history up to t (inclusive)
        w = min(hedge_lookback, t + 1)
        start = t - w + 1

        x = cb[start : t + 1]
        y = ca[start : t + 1]

        # Check for invalid data in window
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if valid_mask.sum() >= 2:
            try:
                # Use numpy.polyfit to get slope (hedge ratio)
                slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)
                # If slope is nan/inf, fallback
                if not np.isfinite(slope):
                    hr = last_hr
                else:
                    hr = float(slope)
            except Exception:
                hr = last_hr
        else:
            # Not enough data to fit - reuse last hedge ratio
            hr = last_hr

        hedge_ratio[t] = hr
        last_hr = hr

        # Compute spread at t
        if np.isfinite(ca[t]) and np.isfinite(cb[t]) and np.isfinite(hr):
            spread[t] = ca[t] - hr * cb[t]
        else:
            spread[t] = np.nan

        # Rolling mean/std for z-score using available data
        wz = min(zscore_lookback, t + 1)
        start_z = t - wz + 1
        window = spread[start_z : t + 1]
        valid_w = window[np.isfinite(window)]
        if valid_w.size >= 1:
            mu = float(valid_w.mean())
            sigma = float(valid_w.std(ddof=0))
            # Avoid division by zero
            if sigma < 1e-12:
                zscore[t] = 0.0
            else:
                if np.isfinite(spread[t]):
                    zscore[t] = (spread[t] - mu) / sigma
                else:
                    zscore[t] = 0.0
        else:
            zscore[t] = 0.0

    # Ensure no NaNs remain by forward-filling any remaining NaNs conservatively
    # (should be unnecessary due to above logic, but keeps outputs clean)
    # For hedge_ratio, replace any leading NaNs with first finite value
    if not np.isfinite(hedge_ratio).all():
        finite_idx = np.where(np.isfinite(hedge_ratio))[0]
        if finite_idx.size > 0:
            first_finite = hedge_ratio[finite_idx[0]]
            hedge_ratio[: finite_idx[0]] = first_finite
        else:
            hedge_ratio[:] = 1.0

    # Replace any remaining NaNs in zscore with 0
    zscore[~np.isfinite(zscore)] = 0.0

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


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
    Order function for vectorbt flexible multi-asset mode.

    This function is called once per asset (column) per bar. The context `c`
    provides:
        - c.i: current bar index
        - c.col: column index (0 for asset A, 1 for asset B)
        - c.position_now: current position (amount) for that asset

    We implement the specified pairs trading logic:
      - Entry when |zscore| > entry_threshold:
          * z > +entry: SHORT A, LONG B (B sized by hedge_ratio)
          * z < -entry: LONG A, SHORT B
      - Exit when z crosses 0.0 (mean reversion) or when |z| > stop_threshold (stop-loss)
      - Position sizing: fixed notional per leg for asset A; asset B sized by hedge_ratio

    Returns:
        (size, size_type, direction)
        - size_type uses vectorbt SizeType numeric mapping (Value=1, TargetAmount=3)
        - direction uses Direction mapping (LongOnly=0, ShortOnly=1, Both=2)
        - To signal 'no order', return (np.nan, 0, 0)
    """
    i = int(getattr(c, "i"))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Safety checks
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else 0.0
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan
    pa = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    pb = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # Numeric tolerance for zero-position check
    tiny = 1e-8
    has_pos = abs(pos_now) > tiny

    # STOP-LOSS: absolute zscore beyond stop_threshold -> close any open position
    if abs(z) > stop_threshold and has_pos:
        # Set target amount to 0 to close regardless of side
        return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # EXIT on mean reversion: z crosses zero
    if i > 0:
        prev_z = float(zscore[i - 1]) if np.isfinite(zscore[i - 1]) else 0.0
        # Detect sign crossing (strict) or crossing to zero
        crossed = (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold)
        if crossed and has_pos:
            return (0.0, SIZE_TYPE_TARGET_AMOUNT, DIRECTION_BOTH)

    # ENTRY logic
    # Only enter if we currently have no position in this asset
    if not has_pos:
        # Long/Short decisions based on z
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                # Asset A: short fixed notional
                if np.isfinite(pa):
                    return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORTONLY)
                else:
                    return (np.nan, 0, 0)
            else:
                # Asset B: sized by hedge ratio: units_B = hr * (notional / pa)
                if np.isfinite(pb) and np.isfinite(pa) and np.isfinite(hr):
                    units_a = notional_per_leg / pa
                    units_b = hr * units_a
                    size_value_b = abs(units_b) * pb
                    if size_value_b <= 0:
                        return (np.nan, 0, 0)
                    direction_b = DIRECTION_LONGONLY if units_b > 0 else DIRECTION_SHORTONLY
                    return (float(size_value_b), SIZE_TYPE_VALUE, direction_b)
                else:
                    return (np.nan, 0, 0)

        elif z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                if np.isfinite(pa):
                    return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONGONLY)
                else:
                    return (np.nan, 0, 0)
            else:
                if np.isfinite(pb) and np.isfinite(pa) and np.isfinite(hr):
                    units_a = notional_per_leg / pa
                    units_b = hr * units_a
                    size_value_b = abs(units_b) * pb
                    if size_value_b <= 0:
                        return (np.nan, 0, 0)
                    direction_b = DIRECTION_SHORTONLY if units_b > 0 else DIRECTION_LONGONLY
                    # Note: When hr>0 and we want to short B here (units_b>0), direction should be Short
                    # If hr<0 the sign flips accordingly.
                    return (float(size_value_b), SIZE_TYPE_VALUE, direction_b)
                else:
                    return (np.nan, 0, 0)

    # Otherwise, no order
    return (np.nan, 0, 0)
