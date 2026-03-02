from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread, and z-score for a pair of assets.

    Args:
        close_a: 1D numpy array of close prices for asset A
        close_b: 1D numpy array of close prices for asset B
        hedge_lookback: lookback window for rolling OLS to compute hedge ratio
        zscore_lookback: window for rolling mean/std of spread to compute z-score

    Returns:
        Dict with keys:
            - 'hedge_ratio': numpy array of rolling hedge ratios (slope of A~B)
            - 'spread': numpy array of spread values Price_A - hedge_ratio * Price_B
            - 'zscore': numpy array of z-score of the spread

    Notes:
        - Uses scipy.stats.linregress for the rolling OLS slope.
        - Handles NaNs by producing NaN outputs where insufficient data is available.
    """
    # Basic validation and conversions
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]
    if n == 0:
        return {"hedge_ratio": np.array([]), "spread": np.array([]), "zscore": np.array([])}

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: regress Price_A ~ Price_B to get slope (hedge ratio)
    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        end = i + 1
        y = close_a[start:end]
        x = close_b[start:end]

        # Skip if any NaNs in window
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x is constant, slope is undefined; set slope to 0 (no hedge)
        if np.allclose(x, x[0]):
            hedge_ratio[i] = 0.0
            continue

        try:
            lr = stats.linregress(x, y)
            hedge_ratio[i] = float(lr.slope)
        except Exception:
            # Fallback to numpy least squares in case linregress fails
            try:
                # Add intercept term
                A = np.vstack([x, np.ones_like(x)]).T
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan

    # Compute spread: Price_A - hedge_ratio * Price_B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score using pandas (handles windows cleanly)
    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    rolling_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(rolling_mean)) & (~np.isnan(rolling_std)) & (rolling_std > 0)
    zscore[valid_z] = (spread[valid_z] - rolling_mean[valid_z]) / rolling_std[valid_z]

    return {"hedge_ratio": hedge_ratio, "spread": spread, "zscore": zscore}


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
    Order function for flexible multi-asset backtest.

    This function is called once per column (asset) per bar by the flexible wrapper.
    It must return a tuple (size, size_type, direction). If size is NaN, the wrapper
    will treat it as NoOrder for that column.

    Orders are specified with SizeType and Direction enums from vectorbt.

    Logic implemented:
      - Entry when zscore > entry_threshold: short A, long B (hedge_ratio units)
      - Entry when zscore < -entry_threshold: long A, short B (hedge_ratio units)
      - Exit when zscore crosses 0.0 (close both legs)
      - Stop-loss when |zscore| > stop_threshold (close both legs)
      - Position sizing: fixed notional_per_leg for Asset A; Asset B sized to maintain
        hedge_ratio units relative to Asset A (notional for B computed accordingly).

    Returns:
        (size, size_type, direction)
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 -> asset_a, 1 -> asset_b

    # Defensive conversions
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    # Default: no order
    NO_ORDER: Tuple[float, int, int] = (np.nan, int(SizeType.Amount), int(Direction.Both))

    # Ensure index in bounds
    if i < 0 or i >= len(zscore):
        return NO_ORDER

    z = zscore[i]
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # If we don't have valid data, do nothing
    if np.isnan(z) or np.isnan(price_a) or np.isnan(price_b) or np.isnan(hr):
        return NO_ORDER

    # Current position (units) for this column
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)
    abs_pos_now = abs(pos_now)
    eps = 1e-8

    # Helper for close order (target value 0)
    close_order: Tuple[float, int, int] = (0.0, int(SizeType.TargetValue), int(Direction.Both))

    # Stop-loss: close if |zscore| > stop_threshold and we have an open position
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold and abs_pos_now > eps:
        return close_order

    # Cross-zero exit: zscore crosses 0.0 (use previous bar)
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev_z = zscore[i - 1]
        crossed = (prev_z < 0 and z >= 0) or (prev_z > 0 and z <= 0) or (z == 0)
        if crossed and abs_pos_now > eps:
            return close_order

    # Entry logic: only enter if we currently hold no position in this column
    if abs_pos_now <= eps:
        # Z-score large positive -> SHORT A, LONG B
        if z > entry_threshold:
            if col == 0:
                # Short Asset A: fixed notional
                return (float(notional_per_leg), int(SizeType.Value), int(Direction.ShortOnly))
            else:
                # Asset B: determine units to match hedge ratio
                # units_A implied by notional_per_leg
                units_a = notional_per_leg / price_a
                units_b = hr * units_a
                if abs(units_b) < eps:
                    return NO_ORDER
                notional_b = abs(units_b * price_b)
                # Direction depends on sign of units_b: positive -> long, negative -> short
                dir_b = int(Direction.LongOnly) if units_b > 0 else int(Direction.ShortOnly)
                return (float(notional_b), int(SizeType.Value), dir_b)

        # Z-score large negative -> LONG A, SHORT B
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (float(notional_per_leg), int(SizeType.Value), int(Direction.LongOnly))
            else:
                units_a = notional_per_leg / price_a
                units_b = hr * units_a
                if abs(units_b) < eps:
                    return NO_ORDER
                notional_b = abs(units_b * price_b)
                dir_b = int(Direction.ShortOnly) if units_b > 0 else int(Direction.LongOnly)
                return (float(notional_b), int(SizeType.Value), dir_b)

    # Otherwise, no order
    return NO_ORDER
