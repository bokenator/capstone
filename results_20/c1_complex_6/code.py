import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Union, Dict


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
    Generate orders for pairs trading. Called by vectorbt's from_order_func (flexible=True).

    Args:
        c: vectorbt OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array with z-score of the spread
        hedge_ratio: 1D array with rolling hedge ratio (slope)
        entry_threshold: threshold to enter a trade (e.g., 2.0)
        exit_threshold: threshold to consider exit (e.g., 0.0)
        stop_threshold: stop-loss threshold to close positions (e.g., 3.0)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
        - size: number of shares to trade (positive = buy, negative = sell)
        - size_type: 0 = Amount (shares)
        - direction: 0 = Both

    Notes:
        - This function is written for flexible multi-asset mode and is called
          once per asset per bar. It calculates the target position for the
          pair and returns the delta required for the current asset.
        - No numba is used.
    """
    i = int(c.i)
    col = int(c.col)

    # Basic bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are numpy and long enough
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(close_a)
    if len(close_b) != n or len(zscore) != n or len(hedge_ratio) != n:
        # Shapes must match
        return (np.nan, 0, 0)

    if i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If any critical value is NaN or non-positive, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute shares per asset according to example in prompt.
    # Note: shares_b is scaled by hedge ratio as in the provided example.
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Previous z for exit cross detection
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine target positions (in shares) for both assets
    target_a = None
    target_b = None

    # Stop-loss: if |z| > stop_threshold then close both
    if not np.isnan(z) and abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry conditions
        if z > entry_threshold:
            # Short A, Long B (scaled by hedge ratio)
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +shares_a
            target_b = -shares_b
        else:
            # Exit on crossing exit_threshold (typically 0.0)
            crossed = False
            if not np.isnan(z_prev):
                # Cross from above to at/below exit_threshold or below to at/above
                if (z_prev > exit_threshold and z <= exit_threshold) or (
                    z_prev < exit_threshold and z >= exit_threshold
                ):
                    crossed = True
            if crossed:
                target_a = 0.0
                target_b = 0.0
            else:
                # No signal to open/close
                return (np.nan, 0, 0)

    # Current position for this asset (ensure numeric)
    pos_now = float(c.position_now) if not (c.position_now is None or np.isnan(c.position_now)) else 0.0

    # Select target for this column
    target = target_a if col == 0 else target_b

    # Compute delta (shares to trade)
    delta = float(target - pos_now)

    # If delta is effectively zero, no action
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return order as amount of shares
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute hedge ratio and z-score for a pairs trading strategy.

    Args:
        asset_a: DataFrame with 'close' column or 1D array of close prices for Asset A
        asset_b: DataFrame with 'close' column or 1D array of close prices for Asset B
        hedge_lookback: lookback period for rolling OLS regression (slope)
        zscore_lookback: lookback period for z-score mean/std

    Returns:
        dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' mapping to 1D numpy arrays

    Notes:
        - Rolling OLS uses scipy.stats.linregress on each window. If the window
          contains NaNs or has zero variance in x, the hedge ratio is left as NaN.
        - Z-score uses pandas rolling mean/std with min_periods equal to lookback.
    """
    # Helper to extract close arrays from inputs
    def _to_close_array(obj) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return obj['close'].astype(float).to_numpy()
        if isinstance(obj, pd.Series):
            return obj.astype(float).to_numpy()
        # Assume array-like (numpy array or list)
        arr = np.asarray(obj, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Input must be 1D array or DataFrame with a close column')
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 1 or zscore_lookback < 1:
        raise ValueError('Lookbacks must be positive integers')

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio: regress A (y) on B (x)
    # Compute slope for each window ending at i (store at i)
    for i in range(hedge_lookback, n + 1):
        # Window is [i-hedge_lookback, i)
        start = i - hedge_lookback
        end = i
        x = close_b[start:end]
        y = close_a[start:end]

        # Skip windows with NaNs or zero variance in x
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        if np.all(x == x[0]):
            # zero variance
            continue
        # Use scipy.stats.linregress to get slope
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratio[end - 1] = float(slope)
        except Exception:
            # In unexpected cases leave as NaN
            continue

    # Compute spread and z-score
    spread = close_a - hedge_ratio * close_b

    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().to_numpy()

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        # Where std is zero or NaN, set zscore to NaN
        zscore[np.isnan(spread_std) | (spread_std == 0)] = np.nan

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
