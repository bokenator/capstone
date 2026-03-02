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
    notional_per_leg: float,
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func (flexible=True).

    Args:
        c: Order context-like object with attributes i, col, position_now, cash_now
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array (1D numpy array)
        hedge_ratio: Hedge ratio array (1D numpy array)
        entry_threshold: entry z-score threshold (positive)
        exit_threshold: exit z-score threshold (usually 0.0)
        stop_threshold: stop-loss z-score threshold (positive)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction) as described in the prompt
    """
    # Defensive casts
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, "position_now", 0.0))

    # Basic bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are long enough
    n = len(zscore)
    if i >= n or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If any critical value is NaN or invalid, do nothing
    if np.isnan(z) or np.isnan(hr) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Small tolerance
    eps = 1e-8

    # Stop-loss: if |z| > stop_threshold, close any open position on this asset
    if abs(z) > stop_threshold:
        if abs(pos) > eps:
            # Close entire position
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on z-score crossing the exit_threshold (commonly 0.0)
    cross_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        z_prev = float(zscore[i - 1])
        # Detect crossing: previous above & current at/below exit OR previous below & current at/above exit
        if (z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold):
            cross_zero = True
    if z == exit_threshold:
        cross_zero = True

    if cross_zero:
        if abs(pos) > eps:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry signals: only enter if |z| > entry_threshold and not beyond stop_threshold
    if z > entry_threshold and abs(z) <= stop_threshold:
        sign_a = -1.0  # Short Asset A
    elif z < -entry_threshold and abs(z) <= stop_threshold:
        sign_a = 1.0  # Long Asset A
    else:
        # No entry signal
        return (np.nan, 0, 0)

    # Compute target sizes in shares
    # Shares for Asset A based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    # Asset B should be scaled by hedge ratio so that position_b ~= -hedge_ratio * position_a
    target_a = sign_a * shares_a
    target_b = -sign_a * hr * shares_a

    # Determine order size (delta) for this asset column
    if col == 0:
        size = target_a - pos
    else:
        size = target_b - pos

    # If very small, treat as no action
    if abs(size) < eps:
        return (np.nan, 0, 0)

    # Return size as absolute number of shares (size_type = 0)
    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for pairs trading strategy.

    Accepts either DataFrames with a 'close' column or 1D numpy arrays / pandas Series.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    All returned values are numpy arrays of the same length as the input series.
    """

    # Helper to extract close prices from different input types
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            arr = x["close"].astype(float).values
        elif isinstance(x, pd.Series):
            arr = x.astype(float).values
        elif isinstance(x, (np.ndarray, list, tuple)):
            arr = np.asarray(x, dtype=float)
        else:
            raise TypeError("asset_a/asset_b must be DataFrame, Series, or array-like")
        if arr.ndim != 1:
            raise ValueError("close prices must be 1-dimensional")
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan)

    # Compute rolling OLS hedge ratio using past data only (no lookahead).
    # We use an expanding window up to hedge_lookback so that early bars also get a slope estimate.
    for i in range(n):
        # window: [start, i] inclusive
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start : i + 1]
        y = close_a[start : i + 1]
        if x.size >= 2:
            # Use scipy.stats.linregress which performs OLS y = slope * x + intercept
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
            except Exception:
                # Fallback: if regression fails (e.g., constant array), fallback to simple ratio
                if np.allclose(x, 0):
                    slope = 1.0
                else:
                    slope = y.mean() / x.mean() if x.mean() != 0 else 1.0
            hedge_ratio[i] = slope
        elif x.size == 1:
            # With single point, approximate slope by ratio if possible, else propagate previous
            if x[0] != 0:
                hedge_ratio[i] = y[0] / x[0]
            elif i > 0 and not np.isnan(hedge_ratio[i - 1]):
                hedge_ratio[i] = hedge_ratio[i - 1]
            else:
                hedge_ratio[i] = 1.0

    # Ensure no remaining NaN in hedge_ratio by forward-filling last known value
    # (this does not introduce lookahead because we only use past values)
    for i in range(n):
        if np.isnan(hedge_ratio[i]):
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 1.0

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (use min_periods = zscore_lookback to adhere to lookback requirement)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use ddof=0 for population std to avoid NaN when window has 1 sample (but min_periods prevents that)
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely (avoid divide by zero)
    zscore = np.full(n, np.nan)
    eps = 1e-8
    valid = (~np.isnan(spread_mean)) & (spread_std > eps)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]
    # If spread_std is (near) zero but mean exists, set zscore to 0.0 (no normalized deviation)
    zero_std = (~np.isnan(spread_mean)) & (spread_std <= eps)
    zscore[zero_std] = 0.0

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }