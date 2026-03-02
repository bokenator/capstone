import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Tuple, Dict


def order_func(
    c,
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
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    Args:
        c: vectorbt OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array of z-score values for the spread
        hedge_ratio: 1D array of rolling hedge ratios (beta)
        entry_threshold: z-score level to enter (e.g., 2.0)
        exit_threshold: z-score level to exit (e.g., 0.0)
        stop_threshold: z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    pos = float(getattr(c, "position_now", 0.0))

    # Safe access to arrays
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Hedge ratio and prices
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan
    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # Validate inputs
    if not np.isfinite(hr) or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)
    if price_a <= 0 or price_b <= 0 or notional_per_leg <= 0:
        return (np.nan, 0, 0)

    # Compute target share sizes per leg (amount = shares)
    shares_a = float(notional_per_leg) / price_a
    # Keep sign of hedge ratio - it determines whether B leg is long or short relative to A
    shares_b = float(notional_per_leg) / price_b * hr

    # Helper: close position for this asset (if any)
    if abs(z) > stop_threshold:
        # Stop-loss: close any existing position
        if abs(pos) > 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit when zscore crosses the exit_threshold (generalized crossing)
    if i > 0 and np.isfinite(zscore[i - 1]):
        z_prev = float(zscore[i - 1])
        # Check crossing of exit_threshold: sign change around the threshold
        if (z_prev - exit_threshold) * (z - exit_threshold) < 0:
            if abs(pos) > 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Entry logic: only enter when we currently have no position on this asset
    # and zscore breaches entry_threshold
    if abs(pos) < 1e-12:
        if z > entry_threshold:
            # Short A, Long B (size in shares). Note shares_b keeps hedge ratio sign.
            if col == 0:
                target = -shares_a
            else:
                target = +shares_b
            order_size = target - pos
            # If effectively zero, do nothing
            if abs(order_size) < 1e-12:
                return (np.nan, 0, 0)
            return (order_size, 0, 0)

        if z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                target = +shares_a
            else:
                target = -shares_b
            order_size = target - pos
            if abs(order_size) < 1e-12:
                return (np.nan, 0, 0)
            return (order_size, 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for the pairs trading strategy.

    Accepts either DataFrame/Series inputs (with a 'close' column/values) or 1D numpy arrays
    representing close prices for Asset A and Asset B. Returns a dictionary with arrays:
    - 'close_a', 'close_b', 'hedge_ratio', 'zscore'

    Args:
        asset_a: DataFrame with 'close' column or 1D numpy array/Series of closes for Asset A
        asset_b: DataFrame with 'close' column or 1D numpy array/Series of closes for Asset B
        hedge_lookback: lookback for rolling OLS regression to compute hedge ratio
        zscore_lookback: lookback for rolling mean/std of spread

    Returns:
        Dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' mapping to numpy arrays
    """
    # Helper to extract close arrays from different input types
    def _extract_close(x):
        # pandas DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("Input DataFrame must contain 'close' column")
            return x["close"].astype(float).values
        # pandas Series
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # numpy array
        if isinstance(x, np.ndarray):
            return x.astype(float)
        # Other sequence-like (e.g., list)
        try:
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass
        raise TypeError("asset_a and asset_b must be pd.DataFrame (with 'close'), pd.Series or 1D numpy array")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Initialize arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    # Rolling OLS regression to compute hedge ratio (slope of regressing A on B)
    for i in range(hedge_lookback, n + 1):
        end = i
        start = i - hedge_lookback
        y = close_a[start:end]
        x = close_b[start:end]
        # Filter valid (finite) pairs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data points to compute regression
            continue
        try:
            slope = stats.linregress(x[mask], y[mask]).slope
            hedge_ratio[end - 1] = float(slope)
        except Exception:
            hedge_ratio[end - 1] = np.nan

    # Compute spread = A - beta * B
    spread = np.full(n, np.nan, dtype=float)
    finite_mask = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[finite_mask] = close_a[finite_mask] - hedge_ratio[finite_mask] * close_b[finite_mask]

    # Rolling mean and std for z-score with full-window requirement
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
