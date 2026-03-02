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

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Current indicators and prices
    z = float(zscore[i]) if i < len(zscore) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan

    # If zscore missing, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Can't trade without valid prices and hedge ratio
    if not (np.isfinite(price_a) and np.isfinite(price_b) and np.isfinite(hr)):
        return (np.nan, 0, 0)

    # Prevent division by zero
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target sizes based on notional per leg
    shares_a = notional_per_leg / price_a
    # We follow the pattern: shares_b = (notional_per_leg / price_b) * hedge_ratio
    shares_b = (notional_per_leg / price_b) * hr

    # Previous z for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Detect crossing of zero: any change of sign or hitting zero from non-zero
    crossing_zero = False
    if np.isfinite(prev_z):
        if prev_z == 0 and z != 0:
            crossing_zero = True
        elif z == 0 and prev_z != 0:
            crossing_zero = True
        elif prev_z * z < 0:
            crossing_zero = True

    # Stop-loss condition
    stop_loss = np.isfinite(z) and (abs(z) > stop_threshold)

    # Determine desired target positions for both assets
    desired_a = None
    desired_b = None

    if stop_loss or crossing_zero:
        # Close both legs
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entry conditions
        if z > entry_threshold:
            # Short A, Long B (hedge_ratio units)
            desired_a = -shares_a
            desired_b = shares_b
        elif z < -entry_threshold:
            # Long A, Short B (hedge_ratio units)
            desired_a = shares_a
            desired_b = -shares_b
        else:
            # No new signal -> keep current position
            # Indicate no action by returning NaN
            return (np.nan, 0, 0)

    # Pick desired for this column
    desired = desired_a if col == 0 else desired_b

    # Sanity check for desired
    if not np.isfinite(desired):
        return (np.nan, 0, 0)

    # Compute order size required to reach desired target (amount of shares)
    order_size = float(desired - pos_now)

    # If order size is effectively zero, do nothing
    if abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Return as amount (shares)
    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close price array from DataFrame/Series/ndarray
    def _extract_close(obj: Any) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame must contain a 'close' column")
            arr = obj['close'].astype(float).values
        elif isinstance(obj, pd.Series):
            arr = obj.astype(float).values
        elif isinstance(obj, np.ndarray):
            arr = obj.astype(float)
        else:
            raise TypeError("asset_a/asset_b must be pandas DataFrame/Series or numpy ndarray")
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS regression to compute hedge ratio (slope) for each window
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue

        x_m = x[mask]
        y_m = y[mask]

        # If x is constant, slope is undefined
        if np.allclose(x_m, x_m[0]):
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_m, y_m)
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan)
    mask_valid = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[mask_valid] = close_a[mask_valid] - hedge_ratio[mask_valid] * close_b[mask_valid]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }