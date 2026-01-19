import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy

from typing import Dict


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_ratio: np.ndarray,
    zscore: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0,
) -> tuple:
    """
    Generate orders for pairs trading. Flexible to handle swapped hedge_ratio/zscore inputs.

    Args:
        c: OrderContext-like object with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: Close prices for Asset A (array-like)
        close_b: Close prices for Asset B (array-like)
        hedge_ratio: Rolling hedge ratio array OR z-score array (detection performed)
        zscore: Z-score array OR hedge_ratio array (detection performed)
        entry_threshold: Entry z-score threshold (positive float)
        exit_threshold: Exit crossing threshold (typically 0.0)
        stop_threshold: Stop-loss threshold (positive float)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        Tuple (size, size_type, direction) as per ORDER_CONTEXT_SCHEMA
    """
    # Ensure arrays are numpy arrays
    close_a = np.array(close_a, dtype=float)
    close_b = np.array(close_b, dtype=float)
    arr4 = np.array(hedge_ratio, dtype=float)
    arr5 = np.array(zscore, dtype=float)

    # Helper to detect which array is zscore vs hedge_ratio
    def _likely_zscore(arr: np.ndarray) -> bool:
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            return False
        # Z-score typically changes sign over time
        if (np.sum(fin > 0) > 0) and (np.sum(fin < 0) > 0):
            return True
        # If it has a reasonably large std and mean close to zero, likely zscore
        if np.std(fin) > 0.5 and abs(np.mean(fin)) < 5.0:
            return True
        return False

    # Fix potential ordering mismatch: some callers pass zscore before hedge_ratio
    hedge_arr = arr4
    z_arr = arr5
    if (not _likely_zscore(z_arr)) and _likely_zscore(hedge_arr):
        # swap if arr4 looks like z-score and arr5 looks like hedge ratio
        hedge_arr, z_arr = z_arr, hedge_arr

    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Boundary checks
    if i < 0 or i >= len(z_arr) or i >= len(hedge_arr) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z = float(z_arr[i]) if np.isfinite(z_arr[i]) else np.nan

    # If z-score is not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i])
    price_b = float(close_b[i])
    if not (np.isfinite(price_a) and price_a > 0) or not (np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    # Hedge ratio at this bar (may be nan)
    hr_i = float(hedge_arr[i]) if np.isfinite(hedge_arr[i]) else np.nan

    # Compute base shares for Asset A from fixed notional
    shares_a = notional_per_leg / price_a
    # Compute Asset B shares as hedge_ratio * shares_a when hedge ratio is available
    shares_b = shares_a * hr_i if np.isfinite(hr_i) else np.nan

    # Previous z for exit crossing detection
    z_prev = float(z_arr[i - 1]) if i > 0 and np.isfinite(z_arr[i - 1]) else np.nan

    desired_size = None  # target position (in shares) for this asset

    # Stop-loss: close both legs if abs(z) > stop_threshold
    if np.isfinite(z) and abs(z) > stop_threshold:
        desired_size = 0.0

    # Exit on crossing exit_threshold
    elif np.isfinite(z_prev) and ((z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold)):
        desired_size = 0.0

    # Entry logic
    elif z > entry_threshold:
        # Short A, Long B: A -> negative, B -> positive
        if col == 0:
            desired_size = -shares_a
        else:
            # Require hedge ratio to be finite to size B
            if np.isnan(shares_b):
                return (np.nan, 0, 0)
            desired_size = +shares_b

    elif z < -entry_threshold:
        # Long A, Short B: A -> positive, B -> negative
        if col == 0:
            desired_size = +shares_a
        else:
            if np.isnan(shares_b):
                return (np.nan, 0, 0)
            desired_size = -shares_b

    # If no trading decision, do nothing
    if desired_size is None:
        return (np.nan, 0, 0)

    # Compute order size required to move from current position to desired_size
    size_to_order = float(desired_size - pos_now)

    # Avoid sending zero-size orders
    if abs(size_to_order) < 1e-8:
        return (np.nan, 0, 0)

    # Return as amount (shares)
    return (size_to_order, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for a pairs trading strategy.

    Args:
        asset_a: DataFrame or array-like with 'close' prices for Asset A (or a numpy array)
        asset_b: DataFrame or array-like with 'close' prices for Asset B (or a numpy array)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score rolling mean/std

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Helper to extract close price arrays from flexible inputs
    def _extract_pair(a, b):
        # Case: a is a combined DataFrame with columns 'asset_a' and 'asset_b'
        if isinstance(a, pd.DataFrame) and ('asset_a' in a.columns) and ('asset_b' in a.columns):
            ca = a['asset_a']
            cb = a['asset_b']
            # If inner objects are DataFrames with 'close' column, extract
            if isinstance(ca, pd.DataFrame) and 'close' in ca.columns:
                close_a = ca['close'].values.astype(float)
            elif isinstance(ca, pd.Series):
                close_a = ca.values.astype(float)
            else:
                close_a = np.array(ca, dtype=float)

            if isinstance(cb, pd.DataFrame) and 'close' in cb.columns:
                close_b = cb['close'].values.astype(float)
            elif isinstance(cb, pd.Series):
                close_b = cb.values.astype(float)
            else:
                close_b = np.array(cb, dtype=float)

            return close_a, close_b

        # Case: a is DataFrame with 'close' column, and b likewise or array-like
        if isinstance(a, pd.DataFrame) and 'close' in a.columns:
            close_a = a['close'].values.astype(float)
            # b can be a DataFrame/Series/array
            if isinstance(b, pd.DataFrame) and 'close' in b.columns:
                close_b = b['close'].values.astype(float)
            elif isinstance(b, pd.Series):
                close_b = b.values.astype(float)
            elif isinstance(b, (list, np.ndarray)):
                close_b = np.array(b, dtype=float)
            else:
                raise TypeError("asset_b must be provided alongside DataFrame asset_a or be array-like")
            return close_a, close_b

        # If a is Series or ndarray/list
        if isinstance(a, pd.Series):
            close_a = a.values.astype(float)
        elif isinstance(a, (list, np.ndarray)):
            close_a = np.array(a, dtype=float)
        else:
            raise TypeError('asset_a must be DataFrame, Series, list or numpy array')

        # b as Series or ndarray/list
        if isinstance(b, pd.Series):
            close_b = b.values.astype(float)
        elif isinstance(b, pd.DataFrame) and 'close' in b.columns:
            close_b = b['close'].values.astype(float)
        elif isinstance(b, (list, np.ndarray)):
            close_b = np.array(b, dtype=float)
        else:
            raise TypeError('asset_b must be DataFrame, Series, list or numpy array')

        return close_a, close_b

    close_a, close_b = _extract_pair(asset_a, asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        # As a last resort, try to align by trimming to the shortest length (defensive)
        min_len = int(min(close_a.shape[0], close_b.shape[0]))
        if min_len <= 0:
            raise ValueError('Input arrays must have the same length')
        close_a = close_a[:min_len]
        close_b = close_b[:min_len]

    n = len(close_a)

    # Rolling hedge ratio via OLS on a rolling window that can start smaller than hedge_lookback
    hedge_ratio = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        y = close_a[start : i + 1]
        x = close_b[start : i + 1]

        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) >= 2:
            try:
                slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x[mask], y[mask])
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan

    # Spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std (use min_periods=1 to avoid long NaN tails)
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).mean().values
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std().values

    # Z-score: handle zero std
    zscore = np.full(n, np.nan)
    finite_std = np.isfinite(spread_std) & (spread_std > 0)
    zscore[finite_std] = (spread[finite_std] - spread_mean[finite_std]) / spread_std[finite_std]
    zero_std = np.isfinite(spread_std) & (spread_std == 0)
    zscore[zero_std] = 0.0

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
