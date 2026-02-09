import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset mode).

    Args:
        c: Order context (provides c.i, c.col, c.position_now, c.cash_now)
        close_a: Array of Asset A close prices
        close_b: Array of Asset B close prices
        zscore: Array of spread z-scores
        hedge_ratio: Array of rolling hedge ratios (slope of A ~ B)
        entry_threshold: Threshold to enter trades (e.g., 2.0)
        exit_threshold: Threshold to exit trades (e.g., 0.0)
        stop_threshold: Stop-loss threshold (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction) tuple understood by the flexible wrapper.
        We return size as a number of shares (size_type=0) representing the
        change in position (desired - current). If no action, return (np.nan, 0, 0)
    """
    i = int(c.i)
    col = int(c.col) if hasattr(c, 'col') else 0
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If we don't have a valid zscore or hedge ratio, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Avoid division by zero or invalid prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Base share sizing: get number of Asset A shares equivalent to notional_per_leg
    shares_a = notional_per_leg / price_a
    # Asset B shares scaled by hedge ratio (can be fractional)
    shares_b = shares_a * hr

    # Determine previous z-score for crossing detection
    prev_z = float(zscore[i - 1]) if i - 1 >= 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit conditions (take precedence)
    stop_loss = abs(z) > stop_threshold
    cross_zero = False
    if not np.isnan(prev_z):
        # Crossing zero (sign change) or exactly zero
        if prev_z * z < 0 or z == 0.0:
            cross_zero = True

    # Determine desired positions (in shares)
    desired_a = None
    desired_b = None

    if stop_loss or cross_zero:
        # Close both legs
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entry logic
        if z > entry_threshold:
            # Short A, Long B
            desired_a = -shares_a
            desired_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            desired_a = +shares_a
            desired_b = -shares_b
        else:
            # No new trading signal; keep current positions
            return (np.nan, 0, 0)

    # Choose which column we're placing an order for
    if col == 0:
        target = desired_a
    else:
        target = desired_b

    # Compute delta (order size) as desired - current
    delta = float(target - pos_now)

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    # Return number of shares to change (size_type=0 -> amount)
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs strategy.

    This function is robust to being passed either DataFrames with a 'close'
    column or raw numpy arrays.

    Args:
        asset_a: DataFrame (or array-like) with 'close' prices for Asset A
        asset_b: DataFrame (or array-like) with 'close' prices for Asset B
        hedge_lookback: Lookback for the rolling OLS regression
        zscore_lookback: Lookback for spread mean/std used in z-score

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Extract close price arrays from DataFrame or accept numpy arrays
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x['close'].values.astype(float)
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # Assume array-like
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Use a conservative minimum number of points for initial regression to avoid
    # excessive NaNs while preserving lookback behavior. We use at least 2 points
    # but prefer 20 if hedge_lookback is larger (param schema min is 20).
    min_periods = max(2, min(hedge_lookback, 20))

    # Rolling OLS: for each time t, fit regression on past up to `hedge_lookback` bars
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        if len(x) >= min_periods:
            # If x is constant, linregress may fail or return nan slope
            try:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Spread and z-score
    spread = close_a - hedge_ratio * close_b

    # Rolling statistics for z-score; require full window to reduce noise
    spread_s = pd.Series(spread)

    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score, handling division by zero
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - spread_mean[valid_mask]) / spread_std[valid_mask]

    # Clip infinite values
    zscore[~np.isfinite(zscore)] = np.nan

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
