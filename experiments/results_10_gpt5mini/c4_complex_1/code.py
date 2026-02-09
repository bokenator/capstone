import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple, Union


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, pd.Series, np.ndarray],
    asset_b: Union[pd.DataFrame, pd.Series, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs trading strategy.

    Accepts either DataFrame/Series with a 'close' column or plain 1D numpy arrays.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    All values are numpy arrays of the same length.
    """

    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        # DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame must contain 'close' column")
            return x['close'].values.astype(float)
        # Series
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # numpy-like
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.astype(float)
        # If shape is (n,1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel().astype(float)
        raise ValueError('Unsupported input type for close prices')

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset A and B must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')

    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    # Rolling OLS regression to estimate hedge ratio (slope of regression of A ~ B)
    # We follow the example: hedge_ratio at time i is computed using window [i-hedge_lookback, i)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # Skip windows with NaNs
        if np.isnan(y).any() or np.isnan(x).any():
            hedge_ratio[i] = np.nan
            continue
        # If constant series or zero variance in x, slope is undefined
        if np.all(x == x[0]):
            hedge_ratio[i] = np.nan
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = slope

    # Spread: A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().to_numpy()

    # Compute z-score safely
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
    # Where std is zero or NaN, set zscore to NaN
    zscore = np.where((np.isnan(spread_std)) | (spread_std == 0), np.nan, zscore)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }


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
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading (flexible multi-asset, non-numba).

    Logic:
    - If zscore is NaN or hedge_ratio NaN -> no action
    - Stop-loss: if |zscore| > stop_threshold -> close both legs (target 0)
    - Exit: if zscore crosses zero (sign change) OR abs(zscore) <= exit_threshold -> close both legs
    - Entry: if zscore > entry_threshold -> short A, long B
             if zscore < -entry_threshold -> long A, short B
    - Position sizing: fixed notional_per_leg per leg

    Returns (size, size_type, direction), where size is number of shares (delta from current position).
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute share amounts based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    # For asset B, scale by hedge ratio (so that B position matches hedge requirement)
    shares_b = (notional_per_leg / price_b) * hr

    # Previous z for zero-cross detection
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if not np.isnan(prev_z) and not np.isnan(z):
        # Detect sign change
        crossed_zero = (prev_z * z) < 0

    # Stop-loss takes precedence
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Exit condition: either explicit threshold around zero or crossing zero
        exit_condition = False
        # If exit_threshold is exactly 0 (default), use zero-cross detection
        if exit_threshold == 0.0:
            if crossed_zero:
                exit_condition = True
        else:
            if abs(z) <= abs(exit_threshold):
                exit_condition = True

        if exit_condition:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short A, Long B
                desired_a = -shares_a
                desired_b = +shares_b
            elif z < -entry_threshold:
                # Long A, Short B
                desired_a = +shares_a
                desired_b = -shares_b
            else:
                # No trading signal -> no change
                return (np.nan, 0, 0)

    # Select the target for the current asset column
    target = desired_a if col == 0 else desired_b

    # Determine delta = desired - current position
    delta = float(target - pos)

    # If delta is effectively zero, do nothing
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    # Return number of shares to trade (amount), allow both long and short
    return (delta, 0, 0)
