from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


def order_func(
    c: Any,
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
    Generate orders for a pairs trading strategy.

    Args:
        c: Order context (must have attributes i, col, position_now, cash_now)
        close_a: Close prices for Asset A (numpy array)
        close_b: Close prices for Asset B (numpy array)
        zscore: Z-score array for the spread
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        Tuple (size, size_type, direction) where:
          - size: float (positive=buy, negative=sell)
          - size_type: int (0=Amount(shares), 1=Value($), 2=Percent)
          - direction: int (0=Both, 1=LongOnly, 2=ShortOnly)

    Notes:
        - This function is intended to be called in flexible multi-asset mode
          (one call per asset per iteration). It returns an order (delta shares)
          to move the current position to the desired target.
        - No numba is used.
    """

    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Safeguard indexing
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio) and not np.isnan(hedge_ratio[i])) else np.nan

    # If zscore is not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if (i < len(close_a)) else np.nan
    price_b = float(close_b[i]) if (i < len(close_b)) else np.nan

    # If prices are invalid, do nothing
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Calculate base share sizes for Asset A (fixed notional per leg)
    shares_a = notional_per_leg / price_a

    # Determine shares for Asset B using hedge ratio (may be nan)
    if not np.isnan(hr):
        shares_b = (notional_per_leg / price_b) * hr
    else:
        # If hedge ratio not available, avoid opening new pair trades
        shares_b = np.nan

    # Previous z for zero-cross detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit conditions
    exit_cross_zero = False
    if not np.isnan(prev_z):
        # If previous z positive and current <= exit_threshold OR previous negative and current >= exit_threshold
        if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
            exit_cross_zero = True

    stop_loss = abs(z) > stop_threshold

    # If stop-loss or zero-crossing, close any existing position in this asset
    if stop_loss or exit_cross_zero:
        if pos != 0.0:
            # Close entire position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry conditions
    enter_short_a_long_b = z > entry_threshold
    enter_long_a_short_b = z < -entry_threshold

    # If hedge ratio is not available, do not enter new trades
    if (enter_short_a_long_b or enter_long_a_short_b) and np.isnan(shares_b):
        return (np.nan, 0, 0)

    # Determine desired positions for each asset
    if col == 0:
        # Asset A
        if enter_short_a_long_b:
            desired = -shares_a
        elif enter_long_a_short_b:
            desired = shares_a
        else:
            return (np.nan, 0, 0)

        delta = desired - pos
        # If already at target (within tiny tolerance), do nothing
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)

        return (float(delta), 0, 0)

    elif col == 1:
        # Asset B
        if enter_short_a_long_b:
            # Long Asset B with magnitude based on hedge ratio
            desired = shares_b
        elif enter_long_a_short_b:
            # Short Asset B with magnitude based on hedge ratio
            desired = -shares_b
        else:
            return (np.nan, 0, 0)

        delta = desired - pos
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)

        return (float(delta), 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score for the spread.

    Args:
        asset_a: DataFrame with 'close' column OR numpy array of close prices for Asset A
        asset_b: DataFrame with 'close' column OR numpy array of close prices for Asset B
        hedge_lookback: lookback for rolling OLS regression (in periods)
        zscore_lookback: lookback for rolling mean/std of spread

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """

    # Extract close arrays from inputs
    if isinstance(asset_a, pd.DataFrame):
        if "close" not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = np.asarray(asset_a["close"], dtype=float)
    elif isinstance(asset_a, np.ndarray):
        close_a = np.asarray(asset_a, dtype=float)
    else:
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, pd.DataFrame):
        if "close" not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = np.asarray(asset_b["close"], dtype=float)
    elif isinstance(asset_b, np.ndarray):
        close_b = np.asarray(asset_b, dtype=float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Hedge ratio: rolling OLS of y=close_a on x=close_b using past window (excludes current bar)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]

        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            continue

        # If x has zero variance, slope will be nan/infinite; handle safely
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            # Leave as NaN on failure
            continue

    # Compute spread using hedge ratio aligned at the same index
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (use pandas rolling for convenience)
    spread_series = pd.Series(spread)

    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - spread_mean[valid_mask]) / spread_std[valid_mask]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
