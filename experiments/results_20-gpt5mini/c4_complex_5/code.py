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

    This function is written for flexible multi-asset mode (vectorbt will call
    it for each asset column). It computes the desired target position for
    Asset A and Asset B based on the z-score and a rolling hedge ratio, then
    returns the difference between target and current position as an order
    (amount of shares).

    Positioning logic:
    - Entry (z > entry_threshold): Short A, Long B (by hedge ratio)
    - Entry (z < -entry_threshold): Long A, Short B (by hedge ratio)
    - Exit: z crosses exit_threshold OR |z| > stop_threshold -> close both

    Sizing:
    - Fixed notional per leg (notional_per_leg). Number of shares for A is
      notional_per_leg / price_a. For B we use (notional_per_leg / price_b) * hedge_ratio
      (this may be signed depending on hedge_ratio sign).

    Returns:
        (size, size_type, direction)
        - size: float (number of shares for size_type=0)
        - size_type: 0 = Amount (shares)
        - direction: 0 = Both
    """
    i = int(c.i)
    col = int(c.col)

    # Safely get current position (shares). If not provided or invalid, assume 0.
    pos_now = getattr(c, 'position_now', 0.0)
    try:
        pos = float(pos_now)
    except Exception:
        pos = 0.0
    if not np.isfinite(pos):
        pos = 0.0

    # Basic bounds check for index
    if i < 0 or i >= len(close_a) or i >= len(close_b) or i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan
    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # If any essential value is NaN, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Avoid division by zero / invalid prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute share sizing based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    # For asset B, scale by hedge ratio (can be signed). This follows the example
    # shares_b = (notional_per_leg / price_b) * hedge_ratio
    shares_b = (notional_per_leg / price_b) * hr

    # Determine if z crossed the exit threshold compared to previous bar
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_exit = False
    if np.isfinite(prev_z):
        # Crossing (prev - exit) * (curr - exit) < 0 indicates crossing the threshold
        try:
            crossed_exit = (prev_z - exit_threshold) * (z - exit_threshold) < 0
        except Exception:
            crossed_exit = False

    # Stop-loss condition
    stop_loss = abs(z) > stop_threshold

    # Determine desired target positions (in shares) for both assets
    desired_a = None
    desired_b = None

    if stop_loss or crossed_exit:
        # Close both positions
        desired_a = 0.0
        desired_b = 0.0
    elif z > entry_threshold:
        # Short A, Long B by hedge ratio
        desired_a = -shares_a
        desired_b = shares_b
    elif z < -entry_threshold:
        # Long A, Short B by hedge ratio
        desired_a = shares_a
        desired_b = -shares_b
    else:
        # No signal -> no order for this bar/asset
        return (np.nan, 0, 0)

    # Determine order for this column as the difference between desired and current
    if col == 0:
        order_size = desired_a - pos
    elif col == 1:
        order_size = desired_b - pos
    else:
        # Unknown column -> no action
        return (np.nan, 0, 0)

    # If order is effectively zero, skip
    if not np.isfinite(order_size) or abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Return amount (number of shares) order
    return (float(order_size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for the pairs strategy.

    This function is flexible with inputs: asset_a/asset_b can be either
    pandas DataFrames (with a 'close' column), pandas Series, or numpy arrays
    of close prices. It returns numpy arrays for use in the order function.

    Returns a dict with keys:
      - 'close_a', 'close_b': price arrays
      - 'hedge_ratio': rolling OLS slope (filled at index i using data from i-hedge_lookback..i-1)
      - 'zscore': (spread - rolling_mean) / rolling_std using zscore_lookback
    """

    # Helper to extract close price array from different input types
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            arr = x['close'].values.astype(float)
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Allow raw list-like
            try:
                arr = np.asarray(x, dtype=float)
            except Exception:
                raise TypeError('asset_a/asset_b must be DataFrame/Series/ndarray or list-like')
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError('Asset price arrays must have the same shape')

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be at least 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be at least 1')

    # Compute rolling hedge ratio using OLS (y = A, x = B)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Mask invalid values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough valid points
            hedge_ratio[i] = np.nan
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        # If x has zero variance, slope undefined
        if np.nanstd(x_valid) == 0:
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread and z-score
    spread = close_a - hedge_ratio * close_b

    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }