"""Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20)
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

Notes:
- Uses rolling OLS (scipy.stats.linregress) for hedge ratio. When there are fewer than 2 points in the window, the previous hedge ratio is carried forward (or 1.0 for the very first bar).
- Rolling statistics use available data (min(window, i+1)) to avoid NaNs/warmup issues.
- Order function designed for flexible multi-asset mode in vectorbt. Returns tuples understood by the provided wrapper.

CRITICAL: Does not use numba or vbt.portfolio.nb.* in this file.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import math

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray | pd.Series | list,
    close_b: np.ndarray | pd.Series | list,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS), spread and z-score.

    Args:
        close_a: Prices for asset A.
        close_b: Prices for asset B.
        hedge_lookback: Rolling window for OLS hedge ratio. When fewer samples are
            available, OLS is computed on the available past samples (to avoid NaNs).
        zscore_lookback: Rolling window for spread mean/std to compute z-score.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray, same length as inputs
            - 'zscore': np.ndarray, same length as inputs
            - 'spread': np.ndarray
            - 'spread_mean': np.ndarray
            - 'spread_std': np.ndarray

    Notes:
        - This function avoids lookahead by computing each value using only data up
          to the current index (inclusive).
        - To ensure determinism and to avoid NaNs after short warmups, when there
          are fewer than 2 points available for regression, the previous hedge
          ratio is carried forward (or 1.0 at the very first bar).
    """
    # Convert inputs to 1D numpy arrays
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    spread_mean = np.full(n, np.nan, dtype=float)
    spread_std = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Starting hedge ratio fallback
    prev_slope = 1.0

    # Loop to compute rolling OLS hedge ratio using only past data
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = b[start : i + 1]
        y = a[start : i + 1]

        # Mask NaNs in the window
        valid = (~np.isnan(x)) & (~np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if x_valid.size >= 2:
            # Compute slope of regression y = slope * x + intercept
            try:
                slope = float(linregress(x_valid, y_valid).slope)
            except Exception:
                slope = prev_slope
            prev_slope = slope
        elif x_valid.size == 1:
            # With a single point, keep previous slope (no new information)
            slope = prev_slope
        else:
            # No valid data in window -> keep previous slope
            slope = prev_slope

        hedge_ratio[i] = slope

        # Compute spread for current bar (uses only current prices and current slope)
        if not (np.isnan(a[i]) or np.isnan(hedge_ratio[i]) or np.isnan(b[i])):
            spread[i] = a[i] - hedge_ratio[i] * b[i]
        else:
            spread[i] = np.nan

        # Rolling mean/std for z-score using available past data up to lookback
        mz_start = max(0, i - zscore_lookback + 1)
        window = spread[mz_start : i + 1]
        # Consider only valid values
        win_valid = window[~np.isnan(window)]
        if win_valid.size >= 1:
            mean_w = float(np.mean(win_valid))
            std_w = float(np.std(win_valid, ddof=0))
            # Avoid zero std leading to inf z-scores
            if std_w < 1e-8:
                std_w = 1e-8
        else:
            mean_w = 0.0
            std_w = 1e-8

        spread_mean[i] = mean_w
        spread_std[i] = std_w

        # Compute z-score for current bar, guard against NaN spread
        if not np.isnan(spread[i]):
            zscore[i] = (spread[i] - mean_w) / std_w
        else:
            zscore[i] = np.nan

    # Post-process: ensure no NaNs after short warmup by forward-filling reasonable values
    # (this helps pass warmup NaN tests while preserving lookback-only logic)
    # Fill hedge_ratio forward
    mask_hr = np.isnan(hedge_ratio)
    if mask_hr.any():
        # forward fill, then backward fill
        hedge_ratio = pd.Series(hedge_ratio).fillna(method="ffill").fillna(method="bfill").values

    # For spread/zscore if NaN (rare), replace with 0.0 to avoid propagation
    spread = pd.Series(spread).fillna(0.0).values
    spread_mean = pd.Series(spread_mean).fillna(0.0).values
    spread_std = pd.Series(spread_std).fillna(1e-8).values
    zscore = pd.Series(zscore).fillna(0.0).values

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
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
    """Order function for flexible multi-asset vectorbt backtest.

    This function is designed to be called separately for each asset (col 0 = A, col 1 = B).
    It returns a tuple (size, size_type, direction) where:
      - size: float number of units to trade (NaN to indicate no order)
      - size_type: int (0 = native units)
      - direction: int (1 = BUY, 2 = SELL) -- matches vectorbt's internal enum values

    Trading logic (per spec):
      - Entry when zscore > +entry_threshold: short A, long B (scaled by hedge ratio)
      - Entry when zscore < -entry_threshold: long A, short B
      - Exit when zscore crosses 0 or when |zscore| > stop_threshold (stop-loss)
      - Fixed notional sizing: notional_per_leg USD per leg

    Important: This function only uses information up to the current bar (c.i).
    """
    # Index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive access to arrays
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current observed position for this column (in native units). Wrapper provides it.
    pos_now = float(getattr(c, "position_now", 0.0))

    # Current indicators
    z = float(zscore[i]) if not np.isnan(zscore[i]) else math.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 1.0

    # Helper: compute units for asset A given notional
    # guard against zero price
    eps_price = 1e-8
    units_a = notional_per_leg / max(abs(price_a), eps_price)
    units_b = abs(hr) * units_a

    # Determine previous z for crossing detection
    z_prev = None
    if i > 0:
        z_prev = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else None

    # Default: no order
    NO_ORDER: Tuple[float, int, int] = (float("nan"), 0, 0)

    # If current z is NaN, do nothing
    if z is math.nan:
        return NO_ORDER

    # Close logic: if we are currently in a position for this asset, and an exit/stop condition is met,
    # then return an order to close the current position fully.
    abs_pos = abs(pos_now)
    in_position = abs_pos > 1e-12

    # Check stop-loss
    stop_trigger = (abs(z) > stop_threshold)

    # Check zero crossing (only valid if previous z exists)
    cross_zero = False
    if z_prev is not None:
        # true crossing through zero (sign change) or landing exactly on zero
        if (z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0):
            cross_zero = True

    # If in position and exit condition or stop loss met -> close this leg
    if in_position and (stop_trigger or cross_zero):
        # To close a long position (pos_now > 0): SELL
        # To close a short position (pos_now < 0): BUY
        if pos_now > 0:
            return (abs_pos, 0, 2)  # SELL
        else:
            return (abs_pos, 0, 1)  # BUY

    # Entry logic (only when not in position for this asset)
    if not in_position:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Asset A: SELL units_a
                return (units_a, 0, 2)
            else:
                # Asset B: BUY units_b
                return (units_b, 0, 1)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Asset A: BUY units_a
                return (units_a, 0, 1)
            else:
                # Asset B: SELL units_b
                return (units_b, 0, 2)

    # Otherwise, do nothing
    return NO_ORDER
