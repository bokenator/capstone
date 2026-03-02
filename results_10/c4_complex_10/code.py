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
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))  # Current position for this asset

    # Basic safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Protect against index out of bounds
    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If any necessary value is NaN, skip
    if not np.isfinite(z) or not np.isfinite(price_a) or not np.isfinite(price_b) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    # Compute shares per leg based on fixed notional
    # shares_a represents the number of Asset A shares corresponding to notional_per_leg
    # shares_b_signed represents the number of Asset B shares (can be negative if hedge ratio negative)
    if price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg / price_a)
    shares_b = float(notional_per_leg / price_b * hr)

    # Previous z-score (for crossing detection)
    prev_z = float(zscore[i - 1]) if i - 1 >= 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Helper: small tolerance to avoid tiny orders
    tol = 1e-8

    # Exit / Stop-loss conditions (close both legs)
    # 1) Stop-loss: absolute z-score exceeds stop_threshold
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        if abs(pos) > tol:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # 2) Exit when z-score crosses exit_threshold (typically 0.0)
    # We treat crossing zero as sign change between previous and current z-score
    if np.isfinite(prev_z) and prev_z * z < 0:
        if abs(pos) > tol:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic
    # When z > entry_threshold: Short A, Long B (scaled by hedge_ratio)
    # When z < -entry_threshold: Long A, Short B (scaled by hedge_ratio)
    if z > entry_threshold:
        # Target positions for this scenario
        target = -shares_a if col == 0 else shares_b
        diff = target - pos
        if abs(diff) <= tol:
            return (np.nan, 0, 0)
        return (float(diff), 0, 0)

    if z < -entry_threshold:
        target = shares_a if col == 0 else -shares_b
        diff = target - pos
        if abs(diff) <= tol:
            return (np.nan, 0, 0)
        return (float(diff), 0, 0)

    # No action for other cases
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a numpy array of close prices
        asset_b: DataFrame with 'close' column for Asset B OR a numpy array of close prices
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame with 'close' column or plain numpy arrays
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.Series):
            close_a = asset_a.values.astype(float)
        else:
            if 'close' not in asset_a.columns:
                raise KeyError("asset_a DataFrame must contain 'close' column")
            close_a = asset_a['close'].astype(float).values
    elif isinstance(asset_a, np.ndarray) or isinstance(asset_a, list):
        close_a = np.asarray(asset_a, dtype=float)
    else:
        raise TypeError("asset_a must be a pandas DataFrame/Series or numpy array-like")

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.Series):
            close_b = asset_b.values.astype(float)
        else:
            if 'close' not in asset_b.columns:
                raise KeyError("asset_b DataFrame must contain 'close' column")
            close_b = asset_b['close'].astype(float).values
    elif isinstance(asset_b, np.ndarray) or isinstance(asset_b, list):
        close_b = np.asarray(asset_b, dtype=float)
    else:
        raise TypeError("asset_b must be a pandas DataFrame/Series or numpy array-like")

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling hedge ratio via OLS regression (A ~ B)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove pairs with NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data to compute regression
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            # In case regression fails for numerical reasons, leave NaN
            hedge_ratio[i] = np.nan

    # Compute spread and z-score
    spread = close_a - hedge_ratio * close_b

    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().to_numpy()

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Replace non-finite values with NaN for cleanliness
    zscore = np.where(np.isfinite(zscore), zscore, np.nan)

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
