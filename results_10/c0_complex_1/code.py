# Auto-generated pairs trading strategy implementation for vectorbt
# Implements compute_spread_indicators and order_func

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: 1D numpy array of close prices for asset A
        close_b: 1D numpy array of close prices for asset B
        hedge_lookback: lookback window for rolling OLS to compute hedge ratio
        zscore_lookback: lookback window for rolling mean/std of the spread

    Returns:
        dict with keys:
            - 'hedge_ratio': numpy array of hedge ratios (same length as inputs)
            - 'zscore': numpy array of z-scores for the spread
            - 'spread': numpy array of spreads (optional)
            - 'spread_mean': rolling mean of spread
            - 'spread_std': rolling std of spread

    Notes:
        - Uses OLS regression (linregress) on the last `hedge_lookback` points to
          estimate the hedge ratio such that spread = A - hedge_ratio * B.
        - Handles NaNs by returning NaN for entries where the regression can't be
          computed (insufficient valid points).
    """
    # Input validation
    if not isinstance(close_a, np.ndarray):
        close_a = np.asarray(close_a)
    if not isinstance(close_b, np.ndarray):
        close_b = np.asarray(close_b)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling hedge ratio using OLS regression of A on B
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        end = i + 1
        window_a = close_a[start:end]
        window_b = close_b[start:end]

        # Require at least 2 valid points to run linregress
        mask = np.isfinite(window_a) & np.isfinite(window_b)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue

        try:
            lr = stats.linregress(window_b[mask], window_a[mask])
            slope = lr.slope
            # If slope is finite, assign, else NaN
            hedge_ratio[i] = float(slope) if np.isfinite(slope) else np.nan
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score using pandas (handles NaNs nicely)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std != 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

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
    """
    Order function for vectorbt flexible multi-asset mode (pairs trading).

    The function is stateless and decides on a single order per asset per bar.

    Signals:
      - If zscore > entry_threshold: SHORT A, LONG B (A: sell, B: buy hedge_ratio * units)
      - If zscore < -entry_threshold: LONG A, SHORT B
      - If zscore crosses 0.0 since previous bar OR abs(zscore) > stop_threshold: close positions

    Position sizing:
      - Base notional is `notional_per_leg` applied to asset A.
      - units_A = notional_per_leg / price_A
      - units_B = hedge_ratio * units_A

    Returns a tuple (size, size_type, direction).
      - size: positive float number of units
      - size_type: 0 -> absolute number of units (SizeType.Size)
      - direction: 1 -> buy (long), 2 -> sell (short)

    To signal "NoOrder", return (np.nan, 0, 0).
    """
    # Extract bar index and column
    i = getattr(c, "i", None)
    col = getattr(c, "col", None)

    # Position now (units). If not provided, assume flat
    position_now = getattr(c, "position_now", None)
    if position_now is None:
        # Try alternative attribute names used by different contexts
        position_now = getattr(c, "position", 0.0)
    # Ensure numeric
    try:
        pos = float(position_now)
    except Exception:
        pos = 0.0

    # Basic validation
    if i is None or col is None:
        # Can't make a decision without bar index and column
        return (np.nan, 0, 0)

    # Fetch prices and indicators for this bar
    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # Guard: price must be positive finite
    if not np.isfinite(price) or price <= 0:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if i < len(zscore) and np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) and np.isfinite(hedge_ratio[i]) else np.nan

    # Determine if z crossed zero this bar (requires previous bar)
    crossed_zero = False
    if i > 0 and np.isfinite(z):
        prev_z = zscore[i - 1]
        if np.isfinite(prev_z):
            if (prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0) or z == 0:
                crossed_zero = True

    # Stop-loss: |z| > stop_threshold
    stop_loss = False
    if np.isfinite(z) and np.abs(z) > stop_threshold:
        stop_loss = True

    # Close condition: cross zero OR stop loss
    close_signal = crossed_zero or stop_loss

    # If we have a closing signal and currently have a position on this asset, close it
    if close_signal and pos != 0.0:
        # Close by sending an order with size equal to current position absolute
        size = abs(pos)
        if size == 0:
            return (np.nan, 0, 0)
        # If currently long (pos > 0): sell -> direction 2
        direction = 2 if pos > 0 else 1
        return (float(size), 0, int(direction))

    # If no valid z or hedge ratio, do not issue new entries
    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    # Entry signals
    # Compute base units for asset A (notional applied to A)
    # Guard division by zero
    units_a = notional_per_leg / price if price > 0 else 0.0

    # For col 0 (asset A)
    if col == 0:
        # Already in position -> do nothing (no scaling/reverse in same bar)
        if pos != 0.0:
            return (np.nan, 0, 0)

        # Short A when z > entry_threshold
        if z > entry_threshold:
            size = units_a
            # Short -> sell -> direction 2
            return (float(size), 0, 2)

        # Long A when z < -entry_threshold
        if z < -entry_threshold:
            size = units_a
            # Long -> buy -> direction 1
            return (float(size), 0, 1)

        return (np.nan, 0, 0)

    # For col 1 (asset B)
    else:
        # Already in position -> do nothing
        if pos != 0.0:
            return (np.nan, 0, 0)

        # If hedge ratio is zero, avoid placing meaningless orders
        if hr == 0.0:
            return (np.nan, 0, 0)

        # units for B scaled by hedge ratio
        size_b = abs(hr) * units_a

        # When z > entry_threshold: LONG B (hedge) while SHORT A
        if z > entry_threshold:
            # Long B -> buy -> direction 1
            return (float(size_b), 0, 1)

        # When z < -entry_threshold: SHORT B (hedge) while LONG A
        if z < -entry_threshold:
            # Short B -> sell -> direction 2
            return (float(size_b), 0, 2)

        return (np.nan, 0, 0)
