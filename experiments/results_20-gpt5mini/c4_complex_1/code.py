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
    """
    # Extract context
    i = int(getattr(c, 'i', 0))
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic bounds / NaN checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If indicators not ready, do nothing
    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]
    if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    # Determine shares based on notional and price
    shares_a = notional_per_leg / price_a
    # Follow provided skeleton: scale B leg by hedge ratio
    shares_b = (notional_per_leg / price_b) * hr

    if not (np.isfinite(shares_a) and np.isfinite(shares_b)):
        return (np.nan, 0, 0)

    # Previous z (for crossing detection)
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # 1) Stop-loss has highest priority: |z| > stop_threshold -> close positions
    if abs(z) > stop_threshold:
        if abs(pos) > 0:
            # Close current asset's position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # 2) Exit on crossing exit_threshold (usually 0.0)
    exited = False
    if np.isfinite(prev_z) and np.isfinite(exit_threshold):
        # Cross detection: prev on one side, current on the other or equal to threshold
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            exited = True

    if exited:
        if abs(pos) > 0:
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # 3) Entry logic
    target_shares = None

    if z > entry_threshold:
        # Short Asset A, Long Asset B
        target_shares = -shares_a if col == 0 else shares_b
    elif z < -entry_threshold:
        # Long Asset A, Short Asset B
        target_shares = shares_a if col == 0 else -shares_b
    else:
        # No action
        return (np.nan, 0, 0)

    # Calculate delta to reach target
    delta = float(target_shares - pos)

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return order in Amount (shares)
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
        asset_a: DataFrame with 'close' column for Asset A OR a numpy array/Series of close prices
        asset_b: DataFrame with 'close' column for Asset B OR a numpy array/Series of close prices
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Support both DataFrame inputs (with 'close') and raw numpy arrays / Series
    if isinstance(asset_a, (np.ndarray, pd.Series)):
        close_a = np.asarray(asset_a, dtype=float)
    elif isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain a 'close' column")
        close_a = asset_a['close'].astype(float).values
    else:
        raise TypeError("asset_a must be a pandas DataFrame, Series, or numpy array")

    if isinstance(asset_b, (np.ndarray, pd.Series)):
        close_b = np.asarray(asset_b, dtype=float)
    elif isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain a 'close' column")
        close_b = asset_b['close'].astype(float).values
    else:
        raise TypeError("asset_b must be a pandas DataFrame, Series, or numpy array")

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Hedge ratio via rolling OLS (slope of regression of A on B)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Ensure lookbacks are sensible
    hl = int(max(1, hedge_lookback))
    zl = int(max(1, zscore_lookback))

    for i in range(hl, n):
        y = close_a[i - hl:i]
        x = close_b[i - hl:i]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread (will be NaN where hedge_ratio is NaN)
    spread = np.full(n, np.nan, dtype=float)
    mask_valid_hr = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[mask_valid_hr] = close_a[mask_valid_hr] - hedge_ratio[mask_valid_hr] * close_b[mask_valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zl, min_periods=zl).mean()
    spread_std = spread_series.rolling(window=zl, min_periods=zl).std()

    zscore = np.full(n, np.nan, dtype=float)
    valid = spread_std.values > 0
    valid = valid & np.isfinite(spread) & np.isfinite(spread_mean.values) & np.isfinite(spread_std.values)
    zscore[valid] = (spread[valid] - spread_mean.values[valid]) / spread_std.values[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
