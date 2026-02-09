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
    """
    i = int(getattr(c, 'i', 0))  # Current bar index
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, 'position_now', 0.0))  # Current position for this asset

    # Basic bounds checks
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # If zscore unavailable, do nothing
    if z is None or np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # Require valid prices
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio at this bar (may be NaN if insufficient data)
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if hr is None:
        hr = np.nan

    # Determine the target share sizes based on fixed notional per leg
    # shares_a is number of Asset A shares to trade. shares_b is scaled by hedge ratio.
    # If hedge ratio is NaN, we cannot size the B leg properly -> do not act.
    if np.isnan(hr):
        # Allow closing if in position even if hr is NaN
        # Check stop / exit conditions first
        prev_z = zscore[i - 1] if i > 0 else np.nan
        # Stop-loss
        if abs(z) > stop_threshold:
            # Close position for this asset
            if pos_now == 0:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)
        # Exit on crossing zero
        if i > 0 and not np.isnan(prev_z) and ((prev_z < 0 and z >= exit_threshold) or (prev_z > 0 and z <= exit_threshold)):
            if pos_now == 0:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Compute share sizes (amounts)
    shares_a = notional_per_leg / price_a
    shares_b = shares_a * hr

    # Small tolerance for float comparisons
    tol = 1e-8

    # Check stop-loss first: if triggered, close positions
    if abs(z) > stop_threshold:
        if abs(pos_now) < tol:
            return (np.nan, 0, 0)
        # Close current position
        return (-pos_now, 0, 0)

    # Check exit on z-score crossing exit_threshold (usually 0)
    prev_z = zscore[i - 1] if i > 0 else np.nan
    if i > 0 and not np.isnan(prev_z):
        # Detect crossing through exit_threshold (0.0 default)
        crossed = (prev_z < 0 and z >= exit_threshold) or (prev_z > 0 and z <= exit_threshold)
        if crossed:
            if abs(pos_now) < tol:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)

    # Entry logic
    # If z > entry_threshold -> Short A, Long B
    if z > entry_threshold:
        if col == 0:
            # Asset A: target short position
            target = -shares_a
        else:
            # Asset B: target long position scaled by hedge ratio
            target = +shares_b

    # If z < -entry_threshold -> Long A, Short B
    elif z < -entry_threshold:
        if col == 0:
            target = +shares_a
        else:
            target = -shares_b

    else:
        # No entry/exit signal
        return (np.nan, 0, 0)

    # Compute order size as delta from current position
    order_size = float(target - pos_now)

    # If the order is effectively zero, do nothing
    if abs(order_size) < tol:
        return (np.nan, 0, 0)

    # Return as amount of shares (size_type=0), allow both directions (direction=0)
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
        asset_a: DataFrame with 'close' column for Asset A (or numpy array/Series)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array/Series)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame with 'close' or numpy arrays / Series
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return x['close'].values.astype(float)
        if isinstance(x, (pd.Series, np.ndarray)):
            return np.asarray(x).astype(float).squeeze()
        raise ValueError("asset_a/asset_b must be DataFrame, Series, or numpy array")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS: for index i, use data from (i - window) to (i - 1) inclusive -> window = min(hedge_lookback, i)
    # Start computing as soon as at least 2 past points are available to avoid long NaN lead-in.
    for i in range(n):
        window = min(hedge_lookback, i)
        if window >= 2:
            start = i - window
            end = i  # exclusive, so uses data up to i-1
            x = close_b[start:end]
            y = close_a[start:end]
            # Skip if any NaNs in window
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            # Require variance in x
            if np.all(x == x[0]):
                # Degenerate, skip
                continue
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                hedge_ratio[i] = float(slope)
            except Exception:
                # In case regression fails for numerical reasons, leave as NaN
                hedge_ratio[i] = np.nan

    # Compute spread using current prices and the hedge ratio computed from past data
    spread = np.full(n, np.nan)
    # Only compute spread where hedge_ratio is available
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std for z-score (causal: uses past and current spread values)
    spread_series = pd.Series(spread)
    # Use min_periods=1 so we have values as soon as spread is available
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use ddof=0 to get 0 for single-sample std (avoid NaN)
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    numer = spread - spread_mean
    # Where std is zero or NaN, set zscore to 0 (no dispersion -> no signal)
    with np.errstate(divide='ignore', invalid='ignore'):
        ztemp = numer / spread_std
    # Replace inf/nan with 0
    ztemp[~np.isfinite(ztemp)] = 0.0
    # Only keep ztemp where spread was valid (hedge_ratio computed)
    zscore[valid_idx] = ztemp[valid_idx]

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }