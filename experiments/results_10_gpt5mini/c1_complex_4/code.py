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
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        zscore: Z-score of spread array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)

    Notes:
        - The function computes the desired target position (in shares) for each
          asset and returns the order size required to reach that target from the
          current position (i.e., desired_position - current_position).
        - If hedge ratio or prices are not available (NaN), the function returns
          no action for that bar/asset.
    """
    i = int(c.i)
    col = int(c.col)

    # Basic bounds & NaN checks
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Read indicators/prices for this bar
    z = zscore[i] if i < len(zscore) else np.nan
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If we cannot compute necessary values, do nothing
    if np.isnan(z) or np.isnan(price_a) or np.isnan(price_b) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Avoid invalid prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base share sizing based on notional per leg.
    # We size Asset A such that dollar exposure is notional_per_leg,
    # and size Asset B according to the hedge ratio (shares_b = shares_a * hedge_ratio).
    shares_a = notional_per_leg / price_a
    shares_b = shares_a * hr

    # Current position for this asset
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Determine desired positions (in shares) for both assets depending on z-score
    # Priority: stop-loss > exit crossing > entries
    desired_a = None
    desired_b = None

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Check crossing of exit_threshold (e.g., 0.0)
        prev_z = zscore[i - 1] if i - 1 >= 0 else np.nan
        crossed_exit = False
        if not np.isnan(prev_z):
            # Crossing occurs when previous and current are on different sides of threshold
            if (prev_z - exit_threshold) * (z - exit_threshold) < 0:
                crossed_exit = True

        if crossed_exit:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short A, Long B
                desired_a = -shares_a
                desired_b = +shares_b
            elif z < -entry_threshold:
                # Long A, Short B
                desired_a = +shares_a
                desired_b = -shares_b
            else:
                # No change requested
                return (np.nan, 0, 0)

    # Map desired position to current column
    desired_pos = desired_a if col == 0 else desired_b

    # Sanity check
    if desired_pos is None:
        return (np.nan, 0, 0)

    # Compute order size needed to reach desired position from current position
    order_size = float(desired_pos - pos_now)

    # If order size is effectively zero, do nothing
    if abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Return order in amount of shares (size_type=0), allow both directions
    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or a numpy array/series of closes)
        asset_b: DataFrame with 'close' column for Asset B (or a numpy array/series of closes)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame with 'close' column or raw arrays/series
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].values.astype(float)
    else:
        # Assume array-like (np.ndarray or pd.Series)
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].values.astype(float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)
    if n == 0:
        raise ValueError('Input price series are empty')

    # Ensure lookbacks are sensible
    hedge_lookback = int(max(2, hedge_lookback))
    zscore_lookback = int(max(2, zscore_lookback))

    # Rolling hedge ratio (OLS slope): y = price_a, x = price_b
    hedge_ratio = np.full(n, np.nan)

    for i in range(hedge_lookback, n + 1):
        # window is [i - hedge_lookback, i)
        start = i - hedge_lookback
        end = i
        x = close_b[start:end]
        y = close_a[start:end]

        # Require at least 2 valid points
        if np.isfinite(x).sum() < 2 or np.isfinite(y).sum() < 2:
            hedge_ratio[i - 1] = np.nan
            continue

        # Exclude any pairs where either is NaN
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i - 1] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i - 1] = slope
        except Exception:
            hedge_ratio[i - 1] = np.nan

    # Align hedge_ratio so that value is available at index i (following the example in the prompt)
    # The loop above wrote slope at index i-1 for window ending at i.

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score; protect division by zero
    zscore = np.full(n, np.nan)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
