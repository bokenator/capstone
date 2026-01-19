import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy.stats
from typing import Any, Dict, Tuple


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading (flexible multi-asset mode).

    Args:
        c: Order context with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: Close prices for Asset A (numpy array)
        close_b: Close prices for Asset B (numpy array)
        zscore: Z-score array for the spread (numpy array)
        hedge_ratio: Rolling hedge ratio array (numpy array)
        entry_threshold: Threshold to enter a trade (e.g., 2.0)
        exit_threshold: Threshold to exit when z-score crosses (e.g., 0.0)
        stop_threshold: Stop-loss threshold (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Hedge ratio must be available
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    if np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or price_a <= 0:
        return (np.nan, 0, 0)
    if not np.isfinite(price_b) or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base shares for Asset A (amount mode)
    shares_a = float(notional_per_leg) / price_a
    # Asset B shares are scaled by hedge ratio (can be fractional)
    shares_b = hr * shares_a

    # Helper: compute previous z (for crossing detection)
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit on stop-loss (force close)
    if np.isfinite(z) and np.abs(z) > stop_threshold:
        if pos == 0.0:
            return (np.nan, 0, 0)
        # Close position entirely
        return (-pos, 0, 0)

    # Exit on crossing the exit_threshold (e.g., 0.0)
    if np.isfinite(prev_z) and np.isfinite(z):
        crossed = (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold)
        # Also consider sign change
        if prev_z * z < 0:
            crossed = True
        if crossed:
            if pos == 0.0:
                return (np.nan, 0, 0)
            return (-pos, 0, 0)

    # Entry conditions
    # If z > entry_threshold: short A, long B
    if z > entry_threshold:
        if col == 0:
            target = -shares_a
        else:
            target = shares_b
        order_size = target - pos
        # If already at target (within tiny tolerance), no action
        if np.abs(order_size) < 1e-8:
            return (np.nan, 0, 0)
        return (float(order_size), 0, 0)

    # If z < -entry_threshold: long A, short B
    if z < -entry_threshold:
        if col == 0:
            target = shares_a
        else:
            target = -shares_b
        order_size = target - pos
        if np.abs(order_size) < 1e-8:
            return (np.nan, 0, 0)
        return (float(order_size), 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute close arrays, rolling hedge ratio (OLS), and spread z-score.

    Args:
        asset_a: DataFrame with 'close' column or numpy array of close prices for Asset A
        asset_b: DataFrame with 'close' column or numpy array of close prices for Asset B
        hedge_lookback: Lookback window (in bars) for rolling OLS
        zscore_lookback: Lookback window for z-score rolling mean/std

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore' (all numpy arrays)
    """
    # Accept either DataFrame with 'close' column or raw numpy arrays
    # Asset A
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].values
    elif isinstance(asset_a, (np.ndarray, list, tuple)):
        close_a = np.array(asset_a)
    else:
        raise TypeError('asset_a must be a pandas DataFrame or numpy array-like')

    # Asset B
    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].values
    elif isinstance(asset_b, (np.ndarray, list, tuple)):
        close_b = np.array(asset_b)
    else:
        raise TypeError('asset_b must be a pandas DataFrame or numpy array-like')

    # If lengths differ (e.g., truncated input), align to the shortest length
    if len(close_a) != len(close_b):
        min_len = min(len(close_a), len(close_b))
        close_a = close_a[:min_len]
        close_b = close_b[:min_len]

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS (slope only). Use past data up to and including current index i.
    for i in range(n):
        # Use up to hedge_lookback previous observations (including current)
        window_length = min(hedge_lookback, i + 1)
        if window_length < 2:
            continue
        window_start = i - window_length + 1
        x = close_b[window_start:i + 1]
        y = close_a[window_start:i + 1]
        # Skip window if contains NaNs or non-finite
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        # Compute OLS slope using scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread using available hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (use pd.Series.rolling with trailing window)
    spread_series = pd.Series(spread)
    rolling_mean_series = pd.Series.rolling(spread_series, window=zscore_lookback).mean()
    rolling_std_series = pd.Series.rolling(spread_series, window=zscore_lookback).std()

    spread_mean = rolling_mean_series.values
    spread_std = rolling_std_series.values

    # Compute z-score, guard against division by zero
    zscore = np.full(n, np.nan)
    valid = (np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0))
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': np.array(close_a, dtype=float),
        'close_b': np.array(close_b, dtype=float),
        'hedge_ratio': hedge_ratio.astype(float),
        'zscore': zscore.astype(float),
    }
