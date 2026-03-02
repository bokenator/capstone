"""
Pairs trading strategy utilities for vectorbt flexible order backtest.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio with lookback (hedge_lookback)
- Z-score from rolling mean/std of spread (zscore_lookback)
- Order function returns (size, size_type, direction) tuples (simple Python types)
  that the provided flexible wrapper converts into vectorbt orders.

CRITICAL: This module does NOT use Numba or vectorbt.nb.* directly.
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
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread.

    Args:
        close_a: 1D array of asset A close prices
        close_b: 1D array of asset B close prices
        hedge_lookback: lookback window for rolling OLS (in bars)
        zscore_lookback: lookback window for z-score mean/std (in bars)

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of same length as inputs with rolling slope (A ~ beta * B)
            - "zscore": np.ndarray of same length with rolling z-score of the spread

    Notes:
        - The hedge_ratio is calculated by regressing A on B on each rolling window.
        - The spread is computed as spread = A - beta * B. If beta is NaN for an index,
          spread and zscore will be NaN at that index.
        - Rolling mean/std use pandas.Series.rolling with min_periods equal to lookback
          to avoid spurious values during warmup.
    """
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Basic sanity checks for lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling OLS: A = beta * B + intercept
    # Use np.linalg.lstsq on each window; skip windows with NaNs or zero variance in B
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        y = a[start_idx : end_idx + 1]
        x = b[start_idx : end_idx + 1]

        # Skip if NaNs in window
        if np.isnan(y).any() or np.isnan(x).any():
            continue

        # Skip if no variation in x (would cause singular matrix)
        if np.nanstd(x) < 1e-12:
            continue

        # Design matrix [x, 1]
        X = np.column_stack((x, np.ones_like(x)))
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta = float(coef[0])
            hedge_ratio[end_idx] = beta
        except Exception:
            # In case lstsq fails for numerical reasons, leave as NaN
            hedge_ratio[end_idx] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid_mask = np.isfinite(hedge_ratio)
    spread[valid_mask] = a[valid_mask] - hedge_ratio[valid_mask] * b[valid_mask]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to be consistent and avoid NaNs for small windows
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Compute z-score
    zscore = (spread_series - roll_mean) / roll_std

    # Convert back to numpy and ensure NaNs where appropriate
    zscore_arr = zscore.to_numpy(dtype=float)
    hedge_ratio_arr = hedge_ratio.astype(float)

    # If roll_std is zero or not finite, set zscore to NaN
    bad_std = ~np.isfinite(roll_std.to_numpy(dtype=float)) | (roll_std.to_numpy(dtype=float) == 0)
    zscore_arr[bad_std] = np.nan

    return {"zscore": zscore_arr, "hedge_ratio": hedge_ratio_arr}


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
    Order function for flexible vectorbt backtest (pairs trading).

    This function follows the requested logic:
    - Entry when zscore > entry_threshold or zscore < -entry_threshold
      (open short/long pair respectively)
    - Exit when zscore crosses zero or when |zscore| > stop_threshold (stop-loss)

    Position sizing:
    - Compute base units for asset A as notional_per_leg / price_A
    - Size for asset B is hedge_ratio * base_units (absolute value used for quantity)

    Returns:
        Tuple (size, size_type, direction)
        - size: float (amount or target amount depending on size_type)
        - size_type: int (vectorbt SizeType enum integer)
            * 0 = Amount (absolute units)
            * 1 = Value (dollar value)
            * 3 = TargetAmount (set target units)
        - direction: int (vectorbt Direction enum integer)
            * 0 = LongOnly
            * 1 = ShortOnly
            * 2 = Both

    The calling wrapper treats a returned size of NaN as NoOrder.
    """
    # Constants matching vectorbt.portfolio.enums.SizeTypeT and DirectionT
    SIZE_AMOUNT = 0
    SIZE_VALUE = 1
    SIZE_TARGET_AMOUNT = 3

    DIRECTION_LONG = 0
    DIRECTION_SHORT = 1
    DIRECTION_BOTH = 2

    # Index and which column (0 = asset_a, 1 = asset_b)
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safety: ensure arrays are numpy and indexable
    close_a_arr = np.asarray(close_a, dtype=float).flatten()
    close_b_arr = np.asarray(close_b, dtype=float).flatten()
    z_arr = np.asarray(zscore, dtype=float).flatten()
    beta_arr = np.asarray(hedge_ratio, dtype=float).flatten()

    n = len(z_arr)
    if i < 0 or i >= n:
        return (float("nan"), 0, DIRECTION_BOTH)

    z_i = float(z_arr[i]) if np.isfinite(z_arr[i]) else np.nan
    beta_i = float(beta_arr[i]) if np.isfinite(beta_arr[i]) else np.nan

    # Current price for this column and both prices for sizing
    price_a = float(close_a_arr[i])
    price_b = float(close_b_arr[i])

    # Current position in this column (units). The wrapper provides position_now
    pos_now = float(getattr(c, "position_now", 0.0))

    # ---------- Exit / Stop conditions take priority ----------
    # Crossing zero: check previous z-score sign change
    crossing_zero = False
    if i > 0 and np.isfinite(z_arr[i - 1]) and np.isfinite(z_i):
        if (z_arr[i - 1] * z_i) < 0:
            crossing_zero = True

    stop_loss = np.isfinite(z_i) and (abs(z_i) > float(stop_threshold))

    if crossing_zero or stop_loss:
        # If there is an open position, issue a target-amount=0 order to close
        if abs(pos_now) > 0:
            return (0.0, SIZE_TARGET_AMOUNT, DIRECTION_BOTH)
        else:
            return (float("nan"), 0, DIRECTION_BOTH)

    # ---------- Entry logic ----------
    # Basic validation
    if not (np.isfinite(z_i) and np.isfinite(beta_i)):
        return (float("nan"), 0, DIRECTION_BOTH)
    if price_a <= 0 or price_b <= 0:
        return (float("nan"), 0, DIRECTION_BOTH)

    # Base units for A based on fixed notional per leg
    base_units_a = float(notional_per_leg) / price_a
    if not np.isfinite(base_units_a) or base_units_a <= 0:
        return (float("nan"), 0, DIRECTION_BOTH)

    # Units for B to hedge (use absolute hedge ratio for size; direction handled separately)
    hedge_units_b = abs(beta_i) * base_units_a

    # Entry: short A / long B when zscore > entry_threshold
    if np.isfinite(z_i) and (z_i > float(entry_threshold)):
        if col == 0:
            # Asset A: short
            # Only issue if not already short
            if pos_now < -1e-8:
                return (float("nan"), 0, DIRECTION_BOTH)
            return (float(base_units_a), SIZE_AMOUNT, DIRECTION_SHORT)
        else:
            # Asset B: long
            if pos_now > 1e-8:
                return (float("nan"), 0, DIRECTION_BOTH)
            return (float(hedge_units_b), SIZE_AMOUNT, DIRECTION_LONG)

    # Entry: long A / short B when zscore < -entry_threshold
    if np.isfinite(z_i) and (z_i < -float(entry_threshold)):
        if col == 0:
            # Asset A: long
            if pos_now > 1e-8:
                return (float("nan"), 0, DIRECTION_BOTH)
            return (float(base_units_a), SIZE_AMOUNT, DIRECTION_LONG)
        else:
            # Asset B: short
            if pos_now < -1e-8:
                return (float("nan"), 0, DIRECTION_BOTH)
            return (float(hedge_units_b), SIZE_AMOUNT, DIRECTION_SHORT)

    # Otherwise, no order
    return (float("nan"), 0, DIRECTION_BOTH)
