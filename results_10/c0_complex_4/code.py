"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (A ~ B) with lookback window.
- Z-score computed from spread with rolling mean/std.
- Entry/Exit/Stop-loss logic implemented in order_func.

CRITICAL: No numba usage.
"""
from typing import Dict, Tuple, Any

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
        close_a: Price series for asset A (1D numpy array).
        close_b: Price series for asset B (1D numpy array).
        hedge_lookback: Window length for rolling OLS regression to estimate hedge ratio.
        zscore_lookback: Window length for rolling mean/std of spread to compute z-score.

    Returns:
        dict with keys:
            - "hedge_ratio": ndarray of hedge ratios (same length as inputs)
            - "zscore": ndarray of z-score values (same length as inputs)
            - "spread": ndarray of spread values (same length as inputs)

    Notes:
        - Hedge ratio is obtained by regressing A = alpha + beta * B and taking beta.
        - Returns NaN for periods where windows are not available or invalid.
    """
    # Ensure numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling OLS for hedge ratio (A ~ B)
    # Use simple OLS slope: beta = cov(B, A) / var(B)
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        win_a = a[start_idx : end_idx + 1]
        win_b = b[start_idx : end_idx + 1]

        # Skip windows with NaNs
        if not (np.isfinite(win_a).all() and np.isfinite(win_b).all()):
            hedge_ratio[end_idx] = np.nan
            continue

        mean_a = win_a.mean()
        mean_b = win_b.mean()

        denom = ((win_b - mean_b) ** 2).sum()
        if denom == 0.0:
            hedge_ratio[end_idx] = np.nan
            continue

        numer = ((win_b - mean_b) * (win_a - mean_a)).sum()
        hedge_ratio[end_idx] = numer / denom

    # Spread = A - beta * B (use hedge_ratio at each bar)
    valid_hr = np.isfinite(hedge_ratio)
    for i in range(n):
        if valid_hr[i] and np.isfinite(a[i]) and np.isfinite(b[i]):
            spread[i] = a[i] - hedge_ratio[i] * b[i]
        else:
            spread[i] = np.nan

    # Rolling mean and std for spread using pandas for convenience
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(roll_mean) & np.isfinite(roll_std) & (roll_std > 0)
    zscore[valid] = (spread[valid] - roll_mean[valid]) / roll_std[valid]

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
    }


# Constants for order encoding. We avoid importing vectorbt enums directly in this module.
SIZE_TYPE_SIZE = 0  # interpret size as absolute number of units/shares
DIRECTION_LONG = 1  # buy
DIRECTION_SHORT = 2  # sell


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
    Order function for a pairs strategy in flexible multi-asset mode.

    This function is designed to work with the provided _wrap_flex_order_func wrapper
    which will call it once per asset (col) per bar.

    Args:
        c: Order context (has attributes i, col, position_now, cash_now (optional)).
        close_a: 1D numpy array of asset A close prices.
        close_b: 1D numpy array of asset B close prices.
        zscore: 1D numpy array of z-score values (same length as close arrays).
        hedge_ratio: 1D numpy array of hedge ratios (same length as close arrays).
        entry_threshold: z-score threshold to enter trades (default 2.0).
        exit_threshold: not used directly; exit on z-score crossing 0.0.
        stop_threshold: absolute z-score for stop-loss (default 3.0).
        notional_per_leg: fixed notional per leg ($) used to size positions.

    Returns:
        Tuple of (size, size_type, direction).
        - size: number of units (float). If np.nan, indicates no order.
        - size_type: integer code for size type (0 = absolute units in this implementation).
        - direction: integer code for direction (1 = buy/long, 2 = sell/short).

    Behavior:
        - Entry when |zscore| > entry_threshold and no current position.
          For z > entry_threshold: Short A, Long B (units scaled by hedge_ratio).
          For z < -entry_threshold: Long A, Short B.
        - Exit when zscore crosses zero (sign change) or |zscore| > stop_threshold -> close positions.

    Notes:
        - Uses absolute share sizing (size_type = 0). Sizes are computed so that the base
          A leg has notional equal to notional_per_leg; B leg units are scaled by hedge_ratio.
        - This function returns simple Python tuples (no numba objects).
    """
    # Extract index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 = asset_a, 1 = asset_b

    # Defensive: ensure arrays are numpy and long enough
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, SIZE_TYPE_SIZE, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # If price or indicators are invalid, do nothing
    if not np.isfinite(price):
        return (np.nan, SIZE_TYPE_SIZE, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # Current position in units for this asset (signed). Expect 0 if flat.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Helper: no order
    NO_ORDER = (np.nan, SIZE_TYPE_SIZE, 0)

    # If indicators are not available, do nothing
    if not np.isfinite(z) or not np.isfinite(hr):
        return NO_ORDER

    # Compute base size in units so that A leg has notional "notional_per_leg"
    # Base units for A (positive number)
    if price == 0:
        return NO_ORDER

    # We'll use the price of the respective asset when computing sizes for that asset.
    # The base used for unit scaling is the price of asset A to keep the 1:hedge_ratio unit ratio.
    price_a = float(close_a[i])
    if not np.isfinite(price_a) or price_a == 0:
        return NO_ORDER

    base_units_a = notional_per_leg / price_a

    # Share rounding: we keep floats, vectorbt accepts fractional sizes for many assets in backtests

    # Entry logic - only if current position in this asset is (approximately) zero
    eps = 1e-8
    is_flat = abs(pos_now) <= eps

    # Determine previous z for crossing detection
    prev_z = None
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = float(zscore[i - 1])

    # Exit conditions (if we have a position)
    if not is_flat:
        # Exit on sign crossing of z (crosses zero) OR stop-loss
        crossed_zero = (prev_z is not None) and (prev_z * z < 0)
        stop_loss = abs(z) > float(stop_threshold)

        if crossed_zero or stop_loss:
            size_to_close = abs(pos_now)
            # If currently long (pos_now > 0), we need to sell to close -> DIRECTION_SHORT
            # If currently short (pos_now < 0), we need to buy to close -> DIRECTION_LONG
            if pos_now > 0:
                return (float(size_to_close), SIZE_TYPE_SIZE, DIRECTION_SHORT)
            else:
                return (float(size_to_close), SIZE_TYPE_SIZE, DIRECTION_LONG)

        # Otherwise hold
        return NO_ORDER

    # If flat, check entry signals
    if is_flat:
        if z > float(entry_threshold):
            # Short A, Long B
            if col == 0:
                # Asset A: short base_units_a
                size = float(base_units_a)
                return (size, SIZE_TYPE_SIZE, DIRECTION_SHORT)
            else:
                # Asset B: long hedge_ratio * base_units_a units
                size = float(abs(hr) * base_units_a)
                return (size, SIZE_TYPE_SIZE, DIRECTION_LONG)

        if z < -float(entry_threshold):
            # Long A, Short B
            if col == 0:
                size = float(base_units_a)
                return (size, SIZE_TYPE_SIZE, DIRECTION_LONG)
            else:
                size = float(abs(hr) * base_units_a)
                return (size, SIZE_TYPE_SIZE, DIRECTION_SHORT)

    return NO_ORDER
