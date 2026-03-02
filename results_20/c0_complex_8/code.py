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
        close_a: Prices for asset A (1D array-like).
        close_b: Prices for asset B (1D array-like).
        hedge_lookback: Window length for rolling OLS to estimate hedge ratio.
        zscore_lookback: Window length for rolling mean/std of the spread used to compute z-score.

    Returns:
        A dictionary with at least the following keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-scores (same length as inputs)
            - 'spread' (optional): np.ndarray of the spread values
            - 'rolling_mean' (optional): rolling mean of spread
            - 'rolling_std' (optional): rolling std of spread

    Notes:
        - Uses scipy.stats.linregress on each rolling window to estimate the slope
          of the regression Price_A ~ slope * Price_B + intercept. The slope is
          the hedge ratio.
        - Handles NaN values by propagating NaNs when windows contain NaNs.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS to compute hedge ratio (slope of A ~ B)
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be positive")

    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        x_win = b[start_idx : end_idx + 1]
        y_win = a[start_idx : end_idx + 1]

        # Skip window if NaNs are present
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue

        # If variance of x is zero, regression slope is undefined
        if np.allclose(x_win, x_win[0]):
            # If x is constant, slope cannot be determined -> leave NaN
            continue

        try:
            res = stats.linregress(x_win, y_win)
            hedge_ratio[end_idx] = float(res.slope)
        except Exception:
            # In case linregress fails for numerical reasons
            hedge_ratio[end_idx] = np.nan

    # Compute spread where hedge_ratio is known
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio) & ~np.isnan(a) & ~np.isnan(b)
    spread[valid_hr] = a[valid_hr] - hedge_ratio[valid_hr] * b[valid_hr]

    # Rolling statistics for z-score
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be positive")

    spread_s = pd.Series(spread)
    rolling_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to compute z-score; this is a common choice
    rolling_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(rolling_mean)) & (~np.isnan(rolling_std)) & (rolling_std != 0)
    zscore[valid_z] = (spread[valid_z] - rolling_mean[valid_z]) / rolling_std[valid_z]

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
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
    """
    Order function for flexible multi-asset mode (pairs trading).

    This function is called once per asset (column) per bar by the flexible
    wrapper. It must return a tuple (size, size_type, direction) where:
      - size: number of units to trade (absolute)
      - size_type: integer code for size type (0 = absolute units)
      - direction: 1 for buy/long, 2 for sell/short

    Returning size = np.nan indicates no order for this column on this call.

    Strategy logic implemented:
      - Entry: zscore > entry_threshold -> SHORT A, LONG B (1*A and hedge_ratio*B scaled to notional)
               zscore < -entry_threshold -> LONG A, SHORT B
      - Exit: zscore crosses exit_threshold (usually 0.0) -> close both legs
      - Stop-loss: |zscore| > stop_threshold -> close both legs
      - Position sizing: scale base unit using notional_per_leg applied to asset A
        i.e. base_units = notional_per_leg / price_A; asset B units = base_units * hedge_ratio

    Notes:
      - c.i is the current index (integer)
      - c.col is the current column (0 for asset_a, 1 for asset_b)
      - c.position_now (if present) is the current position size (in units) for this column
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safety checks
    if i < 0:
        return (np.nan, 0, 0)

    if i >= len(zscore) or i >= len(hedge_ratio) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z_i = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr_i = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    if np.isnan(z_i) or np.isnan(hr_i):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if price_a <= 0 or np.isnan(price_a) or price_b <= 0 or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Current position for this column (units). If not provided, assume flat.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Compute the base number of units for asset A that corresponds to the fixed notional
    try:
        base_units = float(notional_per_leg) / price_a
    except Exception:
        return (np.nan, 0, 0)

    # Determine desired target positions for both assets (units)
    target_a = None
    target_b = None

    # Priority: stop-loss -> exit on z-score crossing -> entry
    if abs(z_i) > stop_threshold:
        # Close both legs
        target_a = 0.0
        target_b = 0.0
    else:
        # Check crossing of exit threshold (e.g., zero crossing). Require previous zscore to be available
        crossed = False
        if i > 0 and not np.isnan(zscore[i - 1]):
            z_prev = float(zscore[i - 1])
            if (z_prev > exit_threshold and z_i <= exit_threshold) or (z_prev < exit_threshold and z_i >= exit_threshold):
                crossed = True

        if crossed:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry conditions
            if z_i > entry_threshold:
                # Short A, Long B
                target_a = -base_units
                target_b = base_units * hr_i
            elif z_i < -entry_threshold:
                # Long A, Short B
                target_a = base_units
                target_b = -base_units * hr_i
            else:
                # No action
                return (np.nan, 0, 0)

    # Select this column's target and compute delta
    target = target_a if col == 0 else target_b

    # If target is None, do nothing
    if target is None:
        return (np.nan, 0, 0)

    # Compute difference between desired target and current position
    delta = target - pos_now

    # If change is negligible, do nothing
    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    # size_type: 0 = absolute units
    size_type = 0

    # direction mapping for vectorbt:
    # 1 = buy (long), 2 = sell (short)
    direction = 1 if delta > 0 else 2

    size = float(abs(delta))

    return (size, int(size_type), int(direction))
