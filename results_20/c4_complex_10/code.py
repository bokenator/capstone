import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Any


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
        - This function is called once per asset (col) for each bar in flexible mode.
        - We compute a target position (in shares) for the current asset based on the
          z-score signal and the hedging ratio. We then return the delta (target - current).
        - Return (np.nan, 0, 0) to indicate no order for this asset at this call.
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety: ensure arrays are numpy and indexable
    try:
        z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    # Require valid z and hedge ratio
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute nominal shares sizes per leg
    # shares_a corresponds to 1 "unit" of asset A sized by notional
    shares_a = float(notional_per_leg / price_a)
    # For asset B we scale by hedge ratio so that we hold approximately hr units of B per unit of A
    shares_b = float((notional_per_leg / price_b) * hr)

    # Determine if we should exit due to stop-loss (absolute z too large)
    if np.abs(z) > stop_threshold:
        # Close current position on this asset (set target to 0)
        target_shares = 0.0
        delta = target_shares - pos
        if np.isclose(delta, 0.0):
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)

    # Determine if we should exit due to crossing zero (use previous z to detect crossing)
    crossed_zero = False
    if i > 0 and i - 1 < len(zscore):
        prev_z = zscore[i - 1]
        if not (np.isnan(prev_z) or np.isnan(z)):
            # Crossing if sign changed
            if prev_z * z < 0:
                crossed_zero = True

    # Also consider a proximity-based exit if abs(z) <= exit_threshold (useful when exit_threshold != 0)
    proximity_exit = (exit_threshold is not None) and (np.abs(z) <= abs(exit_threshold))

    # If crossing zero or within exit threshold, close position
    if crossed_zero or proximity_exit:
        if pos == 0:
            return (np.nan, 0, 0)
        delta = -pos
        if np.isclose(delta, 0.0):
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)

    # Entry logic
    target_shares = None
    if z > entry_threshold:
        # Spread high: short A, long B (scaled by hedge ratio)
        if col == 0:
            target_shares = -shares_a
        else:
            target_shares = shares_b
    elif z < -entry_threshold:
        # Spread low: long A, short B
        if col == 0:
            target_shares = shares_a
        else:
            target_shares = -shares_b
    else:
        # No entry signal and no exit -> no action
        return (np.nan, 0, 0)

    # Compute delta from current position
    delta = float(target_shares - pos)
    # If delta is effectively zero, do nothing
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR a 1D numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame-like inputs or raw numpy arrays
    # Extract close price arrays
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.DataFrame):
            if 'close' not in asset_a.columns:
                raise ValueError("asset_a DataFrame must contain 'close' column")
            close_a = asset_a['close'].values.astype(float)
        else:
            close_a = asset_a.values.astype(float)
    elif isinstance(asset_a, (np.ndarray, list, tuple)):
        close_a = np.asarray(asset_a, dtype=float)
    else:
        # Fallback: try to coerce
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.DataFrame):
            if 'close' not in asset_b.columns:
                raise ValueError("asset_b DataFrame must contain 'close' column")
            close_b = asset_b['close'].values.astype(float)
        else:
            close_b = asset_b.values.astype(float)
    elif isinstance(asset_b, (np.ndarray, list, tuple)):
        close_b = np.asarray(asset_b, dtype=float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    # Ensure equal length
    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("Input series must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS to estimate hedge ratio: regress A on B in each rolling window
    # For each index i we compute slope using the previous `hedge_lookback` observations (exclusive of i)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback, n + 1):
        # window is [i-hedge_lookback, i)
        start = i - hedge_lookback
        end = i
        x = close_b[start:end]
        y = close_a[start:end]

        # Mask NaNs
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) < 2:
            # Not enough data to regress
            if i - 1 < n:
                hedge_ratio[i - 1] = np.nan
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        try:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x_valid, y_valid)
            # Place slope at the last index of the window (i-1)
            hedge_ratio[i - 1] = float(slope)
        except Exception:
            # If regression fails, leave NaN
            hedge_ratio[i - 1] = np.nan

    # Compute spread: spread = A - hedge_ratio * B
    # Use element-wise multiplication; where hedge_ratio is NaN, spread will be NaN
    spread = close_a - hedge_ratio * close_b

    # Compute rolling mean and std for z-score with required minimum periods equal to zscore_lookback
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Where spread_std is zero, set zscore to 0 to avoid inf
    zscore = np.where(np.isfinite(zscore), zscore, np.nan)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
