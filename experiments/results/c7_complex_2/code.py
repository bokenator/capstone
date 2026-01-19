import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy
from typing import Dict, Tuple, Any


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset order func).

    Notes:
    - This function is designed to be called by a flexible vectorbt order wrapper that
      passes arguments in the order: (c, close_a, close_b, zscore, hedge_ratio, entry, exit, stop).
    - Returns a tuple (size, size_type, direction) where size_type=0 means Amount (shares).

    Args:
        c: Order context with attributes: i (index), col (0=A,1=B), position_now (current shares), cash_now
        close_a: 1D numpy array of Asset A close prices
        close_b: 1D numpy array of Asset B close prices
        zscore: 1D numpy array of z-score values
        hedge_ratio: 1D numpy array of rolling hedge ratios
        entry_threshold: enter when zscore > entry_threshold (or < -entry_threshold)
        exit_threshold: exit when zscore crosses exit_threshold (typically 0.0)
        stop_threshold: stop-loss when |zscore| > stop_threshold
        notional_per_leg: fixed notional per leg in dollars (default 10000.0)

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Basic sanity checks
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # If zscore or hedge ratio not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    hedge = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if np.isnan(hedge):
        # Without hedge ratio we cannot size the hedge leg reliably
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]
    if not (np.isfinite(price_a) and np.isfinite(price_b)) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute share sizes (shares of A, shares of B scaled by hedge ratio)
    # Use float shares (vectorbt can handle fractional shares)
    shares_a = notional_per_leg / price_a
    shares_b = float(hedge) * shares_a

    # Stop-loss: if |z| > stop_threshold, close position if any
    if np.abs(z) > stop_threshold:
        if pos != 0.0:
            # Close current position for this asset
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # If not in position, consider entries
    if pos == 0.0:
        # Short A, Long B when z is high
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B (hedge ratio scaled)
                return (shares_b, 0, 0)

        # Long A, Short B when z is low
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B
                return (-shares_b, 0, 0)

        # No entry
        return (np.nan, 0, 0)

    # If already in position, consider exits
    # Exit when z-score crosses zero (sign change) or is within exit_threshold
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if i > 0 and np.isfinite(prev_z):
        crossed_zero = (prev_z * z) < 0.0

    if crossed_zero or (np.abs(z) <= exit_threshold):
        # Close this asset's position
        return (-pos, 0, 0)

    # Otherwise, keep holding
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    The hedge ratio at time t is computed by regressing past observations
    of Asset A (y) on Asset B (x) using scipy.stats.linregress. For each t we use up to
    `hedge_lookback` trailing observations (inclusive of t) but fall back to a smaller
    window when not enough history is available. This avoids lookahead while providing
    early indicator values.

    The spread is: spread_t = price_a_t - hedge_ratio_t * price_b_t
    The z-score is computed using a rolling mean/std of the spread over `zscore_lookback` periods
    with a minimum period of 1 (so early values are available).

    Args:
        asset_a: DataFrame with 'close' column or a 1D numpy array of closes for Asset A
        asset_b: DataFrame with 'close' column or a 1D numpy array of closes for Asset B
        hedge_lookback: lookback window for rolling OLS (max window length)
        zscore_lookback: lookback window for rolling mean/std of spread

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore' each mapped to 1D numpy arrays
    """
    # Accept either DataFrame with 'close' or raw numpy arrays
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise KeyError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].values
    elif isinstance(asset_a, np.ndarray):
        close_a = asset_a
    else:
        raise TypeError("asset_a must be a pandas DataFrame or a numpy array")

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise KeyError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].values
    elif isinstance(asset_b, np.ndarray):
        close_b = asset_b
    else:
        raise TypeError("asset_b must be a pandas DataFrame or a numpy array")

    # If lengths differ (e.g., truncated input), operate on the common prefix to avoid length mismatches
    min_len = min(len(close_a), len(close_b))
    if min_len <= 0:
        raise ValueError("asset_a and asset_b must contain data")

    if len(close_a) != min_len:
        close_a = close_a[:min_len]
    if len(close_b) != min_len:
        close_b = close_b[:min_len]

    n = min_len
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS using up to `hedge_lookback` observations (inclusive of current index)
    for t in range(n):
        # window length: use at most hedge_lookback, but don't exceed available history
        window = hedge_lookback if (t + 1) >= hedge_lookback else (t + 1)
        if window < 2:
            # Need at least 2 points for regression
            continue
        start = t - window + 1
        y = close_a[start:t + 1]
        x = close_b[start:t + 1]

        # Require finite values in the regression window
        if np.sum(np.isfinite(x)) != len(x) or np.sum(np.isfinite(y)) != len(y):
            continue

        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            hedge_ratio[t] = float(slope)
        except Exception:
            hedge_ratio[t] = np.nan

    # Compute spread using current prices and the hedge ratio estimated from past data (no lookahead)
    spread = np.full(n, np.nan)
    finite_idx = np.where(np.isfinite(hedge_ratio))[0]
    for idx in finite_idx:
        if np.isfinite(close_a[idx]) and np.isfinite(close_b[idx]):
            spread[idx] = close_a[idx] - hedge_ratio[idx] * close_b[idx]

    # Rolling mean and std for z-score (allow early values with min_periods=1)
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).mean().values
    # Use ddof=0 to get population std (so early single-value windows give std=0)
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score; when std == 0 but spread is finite, set zscore to 0.0 to avoid NaN/inf
    zscore = np.full(n, np.nan)
    for idx in range(n):
        if not np.isfinite(spread[idx]):
            continue
        if np.isfinite(spread_std[idx]) and spread_std[idx] > 0:
            zscore[idx] = (spread[idx] - spread_mean[idx]) / spread_std[idx]
        else:
            # Insufficient variation -> set to 0.0 (safe neutral value)
            zscore[idx] = 0.0

    return {
        'close_a': np.array(close_a, dtype=float),
        'close_b': np.array(close_b, dtype=float),
        'hedge_ratio': np.array(hedge_ratio, dtype=float),
        'zscore': np.array(zscore, dtype=float),
    }