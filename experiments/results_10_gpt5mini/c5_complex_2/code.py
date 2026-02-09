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
    """
    i: int = int(c.i)  # Current bar index
    col: int = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos: float = float(getattr(c, 'position_now', 0.0))  # Current position for this asset

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    if np.isnan(z):
        # No signal if z-score not available
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio must be available to size B leg
    hedge = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    if np.isnan(hedge):
        return (np.nan, 0, 0)

    # Determine base shares for Asset A based on fixed notional
    shares_a = notional_per_leg / price_a
    # Asset B shares scaled by hedge ratio (can be fractional)
    shares_b = hedge * shares_a

    # Determine target positions (shares) for each asset depending on z-score
    target_a = 0.0
    target_b = 0.0

    # Stop-loss: if breached, close positions
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Exit when z-score crosses exit_threshold (typically 0)
        prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan
        crossed_to_zero = False
        if not np.isnan(prev_z):
            # Detect crossing through exit_threshold (0.0 by default)
            # Crossing when signs change or it exactly hits the threshold
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed_to_zero = True

        if crossed_to_zero:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short Asset A, Long Asset B
                target_a = -shares_a
                target_b = shares_b
            elif z < -entry_threshold:
                # Long Asset A, Short Asset B
                target_a = shares_a
                target_b = -shares_b
            else:
                # No new entry; keep existing positions
                target_a = pos if col == 0 else np.nan  # placeholder: will be computed per-col below
                target_b = pos if col == 1 else np.nan

    # Based on which column we're asked to process, compute the order size (delta)
    if col == 0:
        # For asset A: if target_a is NaN it means 'no change' (keep existing)
        if np.isnan(target_a):
            return (np.nan, 0, 0)
        delta = target_a - pos
    else:
        # Asset B
        if np.isnan(target_b):
            return (np.nan, 0, 0)
        delta = target_b - pos

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return number of shares to trade (size_type=0 -> amount/shares), allow both directions
    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or a 1d numpy array/Series)
        asset_b: DataFrame with 'close' column for Asset B (or a 1d numpy array/Series)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close prices from different input types
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return x['close'].to_numpy(dtype=float)
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float)
        # Assume array-like
        arr = np.asarray(x, dtype=float)
        # If a 2D array is passed, attempt to flatten if one column
        if arr.ndim > 1:
            # If it's a 2D with shape (n,1) or (1,n), flatten
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.shape[0] == 1:
                arr = arr[0, :]
            else:
                raise ValueError("Unsupported array shape for close prices")
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset arrays must have the same length")

    n = len(close_a)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling (or expanding for initial bars) OLS hedge ratio (slope)
    # Use only past data up to current index (inclusive) to avoid lookahead
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        y = close_a[start:i + 1]
        x = close_b[start:i + 1]

        # Remove non-finite values within the window
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            # Not enough data to compute regression yet
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using the hedge ratio (element-wise)
    # Wherever hedge_ratio is NaN, the spread will be NaN
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for spread to compute z-score (use past values only)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to be stable
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std != 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    # Return arrays
    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
