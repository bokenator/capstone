"""
Pairs trading strategy: compute spread indicators and flexible order function for vectorbt backtest.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold=2.0, exit_threshold=0.0, stop_threshold=3.0, notional_per_leg=10000.0) -> tuple[float, int, int]

Notes:
- Rolling OLS hedge ratio (A ~ B) computed using up to `hedge_lookback` past points. Early bars use smaller windows (expanding) so there are no long NaN tails.
- Z-score uses rolling mean/std of spread with `zscore_lookback` (min_periods=1). std=0 is handled to avoid divide-by-zero.
- Order function is for vectorbt flexible multi-asset mode. It returns (size, size_type, direction) or (np.nan, 0, 0) for NoOrder.

CRITICAL: This implementation avoids numba and vbt-specific enums in the user code.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert various input types to a 1D numpy float array.

    Accepts: pd.Series, pd.DataFrame, list-like, numpy arrays.
    For DataFrame, returns the first column (or the 'close' column if present).
    """
    # Pandas DataFrame
    if isinstance(x, pd.DataFrame):
        # Prefer a 'close' column if available
        if 'close' in x.columns:
            ser = x['close']
            return pd.Series(ser).astype(float).to_numpy()
        # Otherwise return the first column
        return pd.Series(x.iloc[:, 0]).astype(float).to_numpy()

    # Pandas Series
    if isinstance(x, pd.Series):
        return x.astype(float).to_numpy()

    # Numpy array or other sequence
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # If single column 2D array, flatten
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        # If 2D with multiple columns, return first column
        return arr[:, 0].astype(float)

    # Fallback
    raise ValueError("Unsupported price input shape/type")


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (1D array-like or pandas Series/DataFrame).
        close_b: Prices for asset B (1D array-like or pandas Series/DataFrame).
        hedge_lookback: Maximum lookback window for rolling OLS. Early bars use smaller windows (expanding).
        zscore_lookback: Window for rolling mean/std of the spread used to compute z-score.

    Returns:
        dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-scores (same length as inputs)
            - 'spread': np.ndarray of spread values (same length as inputs)

    Notes:
        - Implementation ensures no lookahead: values at index t are computed using data up to and including t.
        - To avoid NaNs after short warmup, the rolling OLS uses expanding windows up to `hedge_lookback`.
    """
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)
    if n == 0:
        return {"hedge_ratio": np.array([]), "zscore": np.array([]), "spread": np.array([])}

    # Ensure lookbacks are sensible
    hedge_lookback = max(1, int(hedge_lookback))
    zscore_lookback = max(1, int(zscore_lookback))

    hedge_ratio = np.empty(n, dtype=float)
    hedge_ratio[:] = np.nan

    # Rolling OLS (A ~ B) using up to hedge_lookback past points (expanding up to that limit)
    prev_slope = 0.0
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        xa = a[start : i + 1]
        xb = b[start : i + 1]
        mask = (~np.isnan(xa)) & (~np.isnan(xb))
        if mask.sum() >= 2:
            # Ensure variance in xb (denominator) to avoid degenerate regression
            if np.nanstd(xb[mask]) > 0:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(xb[mask], xa[mask])
                    if np.isfinite(slope):
                        hedge_ratio[i] = float(slope)
                        prev_slope = hedge_ratio[i]
                    else:
                        hedge_ratio[i] = prev_slope
                except Exception:
                    # Fallback to previous slope on any regression error
                    hedge_ratio[i] = prev_slope
            else:
                # No variance in xb: reuse previous slope
                hedge_ratio[i] = prev_slope
        else:
            # Not enough points: reuse previous slope (expanding behavior)
            hedge_ratio[i] = prev_slope

    # Compute spread using current hedge ratio (element-wise)
    spread = a - hedge_ratio * b

    # Rolling mean/std for spread -> z-score
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use population std (ddof=0) to be stable for small windows
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Avoid division by zero - where std is very small, set zscore to 0.0
    eps = 1e-8
    zscore = np.zeros(n, dtype=float)
    mask_valid = ~np.isnan(spread) & (roll_std > eps)
    zscore[mask_valid] = (spread[mask_valid] - roll_mean[mask_valid]) / roll_std[mask_valid]
    # For remaining (std == 0 or spread NaN), keep zscore at 0.0

    return {"hedge_ratio": hedge_ratio, "zscore": zscore, "spread": spread}


# Simple enum-like integers for order function
SIZE_TYPE_SIZE = 0  # interpret 'size' as number of units
DIRECTION_LONG = 1
DIRECTION_SHORT = 2


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
    Flexible order function for vectorbt in multi-asset mode.

    Arguments correspond to vbt.Portfolio.from_order_func flexible signature wrapper used in the test harness.

    Returns a tuple (size, size_type, direction), or (np.nan, 0, 0) for NoOrder.

    Logic:
    - Entry when |zscore| > entry_threshold:
        * zscore > threshold -> Short A, Long B
        * zscore < -threshold -> Long A, Short B
      Position sizing: fixed notional per leg (notional_per_leg). A units = notional_per_leg / price_a.
      B units = hedge_ratio * A_units (may be positive or negative); direction set by sign.

    - Exit when zscore crosses zero (sign change) OR |zscore| > stop_threshold -> close positions.

    Notes:
    - c has attributes: i (bar index), col (0 for asset_a, 1 for asset_b), position_now (current position in units), cash_now/value_now
    - All decisions use only information up to index i (no lookahead).
    """
    i = int(getattr(c, "i"))
    col = int(getattr(c, "col", 0))

    # Defensive bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If price is invalid, no action
    if not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    # Current indicators
    z = float(zscore[i]) if (i >= 0 and i < len(zscore)) else np.nan
    hr = float(hedge_ratio[i]) if (i >= 0 and i < len(hedge_ratio)) else np.nan

    # Current position for this column (in units)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Tolerances
    tiny = 1e-8

    # Helper to check if we are effectively flat
    def is_flat(x: float) -> bool:
        return abs(x) < tiny

    # Determine if we need to close (stop-loss or crossing zero)
    stop_loss = np.isfinite(z) and abs(z) > float(stop_threshold)

    crossed_zero = False
    if i > 0 and np.isfinite(zscore[i - 1]) and np.isfinite(z):
        # crossing zero if signs differ (exclude touching zero to avoid ambiguous sign)
        if (zscore[i - 1] < 0 and z > 0) or (zscore[i - 1] > 0 and z < 0):
            crossed_zero = True

    # If we're in a position and need to close, issue closing order for this column
    if not is_flat(pos_now) and (stop_loss or crossed_zero):
        size = abs(pos_now)
        # If currently long (>0), close by selling (SHORT direction). If short (<0), close by buying (LONG direction).
        direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
        return (float(size), SIZE_TYPE_SIZE, int(direction))

    # Entry logic: only if flat on this column
    if is_flat(pos_now) and np.isfinite(z) and np.isfinite(hr) and not np.isnan(z):
        if abs(z) > float(entry_threshold):
            # Units for asset A (absolute magnitude)
            # Avoid division by zero in price
            if price_a <= 0 or price_b <= 0:
                return (np.nan, 0, 0)

            units_a_abs = float(notional_per_leg / price_a)
            # Target sign for A: short when z>0, long when z<0 -> position_a_target = -sign(z) * units_a_abs
            s = np.sign(z) if z != 0 else 1.0
            pos_a_target = -s * units_a_abs

            # Target for B: position_b_target = - position_a_target * hedge_ratio
            pos_b_target = -pos_a_target * hr

            if col == 0:
                # Asset A order
                size = abs(pos_a_target)
                if is_flat(size) or not np.isfinite(size):
                    return (np.nan, 0, 0)
                direction = DIRECTION_LONG if pos_a_target > 0 else DIRECTION_SHORT
                return (float(size), SIZE_TYPE_SIZE, int(direction))
            else:
                # Asset B order
                size = abs(pos_b_target)
                if is_flat(size) or not np.isfinite(size):
                    # If hedge ratio is zero then no B leg
                    return (np.nan, 0, 0)
                direction = DIRECTION_LONG if pos_b_target > 0 else DIRECTION_SHORT
                return (float(size), SIZE_TYPE_SIZE, int(direction))

    # No order for this column at this bar
    return (np.nan, 0, 0)
