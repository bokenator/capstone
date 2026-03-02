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
        close_a: Close prices for Asset A (np.ndarray)
        close_b: Close prices for Asset B (np.ndarray)
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
        - The function computes target positions (in shares) for both assets based
          on the z-score and hedge ratio. It returns the delta between target and
          current position for the asset being processed (c.col).
        - Returns (np.nan, 0, 0) when no action is required.
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    hedge = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    if np.isnan(hedge):
        return (np.nan, 0, 0)

    # Base shares for Asset A derived from fixed notional per leg
    shares_a = notional_per_leg / price_a

    # Shares for Asset B determined by hedge ratio relative to Asset A units
    # i.e., if we trade X shares of A, we trade hedge_ratio * X shares of B
    shares_b = hedge * shares_a

    # Previous z for crossing detection
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit conditions
    stop_condition = np.abs(z) > stop_threshold
    cross_condition = False
    if not np.isnan(z_prev):
        # Detect crossing of exit_threshold (commonly 0.0)
        if (z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold):
            cross_condition = True

    # Determine targets
    target_a = None
    target_b = None

    if stop_condition or cross_condition:
        # Close both legs
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry logic
        if z > entry_threshold:
            # Short Asset A, Long Asset B
            target_a = -shares_a
            target_b = shares_b
        elif z < -entry_threshold:
            # Long Asset A, Short Asset B
            target_a = shares_a
            target_b = -shares_b
        else:
            # No trade signal
            return (np.nan, 0, 0)

    # Select target for this column
    target = target_a if col == 0 else target_b

    # Compute delta to reach target
    delta = float(target - pos)

    # If delta is essentially zero, do nothing
    if np.isclose(delta, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return size in amount (shares), allow both long and short (direction=0)
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
    # Accept either DataFrame/Series with 'close' or raw numpy arrays
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.DataFrame):
            if 'close' not in asset_a:
                raise KeyError("asset_a DataFrame must contain 'close' column")
            close_a = np.asarray(asset_a['close'], dtype=float)
        else:
            close_a = np.asarray(asset_a, dtype=float)
    else:
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.DataFrame):
            if 'close' not in asset_b:
                raise KeyError("asset_b DataFrame must contain 'close' column")
            close_b = np.asarray(asset_b['close'], dtype=float)
        else:
            close_b = np.asarray(asset_b, dtype=float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = close_a.shape[0]

    # Validate lookbacks
    if hedge_lookback < 1:
        raise ValueError("hedge_lookback must be >= 1")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio (slope of regression y = slope * x + intercept), where y = A, x = B
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove NaNs within the window
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < max(2, int(hedge_lookback * 0.5)):
            # Not enough data points to compute a reliable regression
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use ddof=0 for population std to avoid NaN when only one value (but min_periods prevents that)
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score, handle zero std
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }


# Expose functions for import
__all__ = ['compute_spread_indicators', 'order_func']
