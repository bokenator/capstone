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

    This implementation is flexible (multi-asset) and does NOT use numba.

    Args:
        c: Order context with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: Close prices for Asset A (array-like)
        close_b: Close prices for Asset B (array-like)
        zscore: Precomputed z-score array
        hedge_ratio: Precomputed rolling hedge ratio array
        entry_threshold: Entry z-score threshold
        exit_threshold: Exit z-score threshold (usually 0.0)
        stop_threshold: Stop-loss threshold (absolute z-score)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Basic validation / bounds
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio must be available to size leg B
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    if np.isnan(hr):
        return (np.nan, 0, 0)

    # Compute share sizes (may be fractional). Use signed hedge_ratio so direction of leg B
    shares_a = float(notional_per_leg) / price_a
    shares_b = (float(notional_per_leg) / price_b) * hr

    # Previous z for detecting zero-cross
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Tolerance for treating floats as equal
    tol = 1e-8

    # 1) Stop-loss: if |z| > stop_threshold -> close both positions
    if np.isfinite(z) and abs(z) > float(stop_threshold):
        if abs(pos) > tol:
            # Close full position for this asset
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # 2) Exit: if z crosses zero (sign change) or |z| <= exit_threshold -> close
    crossed_zero = False
    if not np.isnan(prev_z):
        if prev_z * z < 0:
            crossed_zero = True

    if crossed_zero or abs(z) <= abs(float(exit_threshold)):
        if abs(pos) > tol:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # 3) Entry logic
    target_a = 0.0
    target_b = 0.0

    if z > float(entry_threshold):
        # Short Asset A, Long Asset B (hedge_ratio units)
        target_a = -shares_a
        target_b = shares_b
    elif z < -float(entry_threshold):
        # Long Asset A, Short Asset B (hedge_ratio units)
        target_a = shares_a
        target_b = -shares_b
    else:
        # No entry/exit signal: do nothing
        return (np.nan, 0, 0)

    # Determine target for this column
    target = target_a if col == 0 else target_b

    # If current position already close to target, do nothing
    if abs(pos - target) <= tol:
        return (np.nan, 0, 0)

    # Generate order to move from current pos to target (in shares)
    size = float(target - pos)

    # Return as shares (size_type=0), allow both directions (direction=0)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for the pairs strategy.

    Supports asset inputs as either pandas DataFrame/Series with a 'close' column
    or as numpy arrays / array-like of close prices.

    Returns a dict with numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    """
    # Extract close arrays from inputs (allow flexible types)
    def _extract_close(x):
        # If DataFrame, only access 'close' as per DATA_SCHEMA
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x["close"].astype(float).values
        # If Series or ndarray or list-like
        if isinstance(x, pd.Series):
            return x.astype(float).values
        if isinstance(x, np.ndarray):
            return x.astype(float)
        # Fallback
        return np.asarray(x, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError("Asset A and Asset B must have the same length")

    n = len(close_a)

    # Initialize hedge ratio with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no lookahead): for index i we use window [i-hedge_lookback, i)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]
        # Require at least 2 non-NaN observations
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio (elementwise). Where hedge_ratio is NaN -> spread NaN
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score. Use min_periods = zscore_lookback to avoid premature values
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Z-score: handle division by zero
    zscore = np.full(n, np.nan, dtype=float)
    ok = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[ok] = (spread[ok] - spread_mean[ok]) / spread_std[ok]

    return {
        "close_a": np.asarray(close_a, dtype=float),
        "close_b": np.asarray(close_b, dtype=float),
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }