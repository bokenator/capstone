import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


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
) -> tuple:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset mode).

    Notes:
    - This function is called once per asset (col) with a simulated context `c`.
    - The wrapper used by the backtest will call this function for both assets each bar.

    Args:
        c: OrderContext-like object with attributes i (bar index), col (asset column),
           position_now (current position in shares), cash_now (available cash) - flexible simulated.
        close_a: 1D numpy array of close prices for Asset A
        close_b: 1D numpy array of close prices for Asset B
        zscore: 1D numpy array of spread z-score values
        hedge_ratio: 1D numpy array of rolling hedge ratio values
        entry_threshold: threshold to enter trades (e.g., 2.0)
        exit_threshold: threshold to exit trades (e.g., 0.0)
        stop_threshold: threshold to force-close trades (e.g., 3.0)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
        - size: number of shares to trade (positive buy, negative sell)
        - size_type: 0 = Amount (shares)
        - direction: 0 = Both

    Behavior:
        - Entry (z > entry_threshold): Short A, Long B (hedge_ratio units)
        - Entry (z < -entry_threshold): Long A, Short B (hedge_ratio units)
        - Exit when z crosses exit_threshold (e.g., 0.0) or |z| > stop_threshold
        - Position sizing uses fixed notional_per_leg per leg
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))

    # Bounds check for arrays
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # If z is not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Current prices
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    # Current position (shares)
    pos_now = float(getattr(c, "position_now", 0.0))
    if not np.isfinite(pos_now):
        pos_now = 0.0

    # Previous z-score (for crossing detection)
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Determine target positions (in shares) for both assets
    target_a = None
    target_b = None

    # Stop-loss: close both legs if |z| > stop_threshold
    if np.isfinite(z) and np.abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Exit when z crosses the exit_threshold (e.g., 0.0)
        crossed_exit = False
        if i > 0 and np.isfinite(prev_z):
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed_exit = True

        if crossed_exit:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry signals
            if z > entry_threshold:
                # Short A (1 unit), Long B (hedge_ratio units)
                # Need valid prices and hedge ratio to size legs
                hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
                if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0 and np.isfinite(hr)):
                    return (np.nan, 0, 0)
                shares_a = float(notional_per_leg) / float(price_a)
                shares_b = (float(notional_per_leg) / float(price_b)) * float(hr)
                target_a = -shares_a
                target_b = shares_b
            elif z < -entry_threshold:
                # Long A (1 unit), Short B (hedge_ratio units)
                hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
                if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0 and np.isfinite(hr)):
                    return (np.nan, 0, 0)
                shares_a = float(notional_per_leg) / float(price_a)
                shares_b = (float(notional_per_leg) / float(price_b)) * float(hr)
                target_a = shares_a
                target_b = -shares_b
            else:
                # No actionable signal
                return (np.nan, 0, 0)

    # Select target for this column
    if col == 0:
        target = target_a
    else:
        target = target_b

    # If target is not finite, do nothing
    if target is None or not np.isfinite(target):
        return (np.nan, 0, 0)

    # Compute order size as difference between target and current position
    size = float(target - pos_now)

    # If size is effectively zero, skip
    if not np.isfinite(size) or abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return as number of shares (Amount)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and spread z-score for a pair of assets.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a 1D numpy array / pd.Series
        asset_b: DataFrame with 'close' column for Asset B OR a 1D numpy array / pd.Series
        hedge_lookback: lookback window for rolling OLS regression (slope)
        zscore_lookback: lookback window for z-score mean/std

    Returns:
        Dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' (all numpy arrays)
    """

    # Helper to extract close price arrays from different input types
    def _extract_close(obj):
        # pandas DataFrame with 'close' column
        if isinstance(obj, pd.DataFrame):
            if "close" not in obj.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return obj["close"].values.astype(float)
        # pandas Series
        if isinstance(obj, pd.Series):
            return obj.values.astype(float)
        # numpy array or other sequence
        if isinstance(obj, np.ndarray):
            return obj.astype(float)
        # Fallback: try to create numpy array
        return np.array(obj, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = close_a.shape[0]

    # Rolling OLS hedge ratio (slope of regression of A ~ B)
    hedge_ratio = np.full(n, np.nan, dtype=float)
    # Compute slope for each rolling window ending at i (value stored at index i)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]
        # Require finite observations
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            # Not enough valid points to run regression
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread using rolling hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
