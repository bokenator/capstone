import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> tuple:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset, no numba).

    This function is called once per asset per bar by the flexible wrapper. It
    computes the desired target position for the current asset (Asset A or B)
    based on the z-score and hedge ratio and returns an order expressed as the
    difference between target position and current position (i.e., the trade
    size in number of shares).

    Size type: 0 (Amount - shares). Direction: 0 (Both allowed).

    Notes:
    - Uses fixed notional sizing per leg as described in the prompt.
    - For Asset B the hedge ratio (which may be negative) is applied to the
      computed shares to produce the target share exposure for hedging.
    - Exits occur when the z-score crosses the exit_threshold OR when the
      absolute z-score exceeds stop_threshold (stop-loss).

    Returns:
        (size, size_type, direction)
    """

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Basic validation
    if i < 0:
        return (np.nan, 0, 0)

    # Guard array bounds
    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If we don't have a valid z-score or hedge ratio, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # Price validation
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute number of shares based on fixed notional per leg
    # Note: shares_b includes hedge ratio to reflect the "hedge_ratio units" requirement
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Check for exit conditions: z-score crossing the exit_threshold or stop-loss
    close_signal = False
    prev_z = np.nan
    if i > 0:
        prev_z = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else np.nan

    # Crossing exit_threshold (e.g., 0.0)
    if not np.isnan(prev_z):
        # Cross from above to below or below to above relative to exit_threshold
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            close_signal = True

    # Stop-loss if zscore magnitude exceeds stop_threshold
    if abs(z) > stop_threshold:
        close_signal = True

    # If close signal, close the current asset position (if any)
    if close_signal:
        if abs(pos_now) < 1e-12:
            return (np.nan, 0, 0)
        # Return order that brings position to zero
        return (-pos_now, 0, 0)

    # Entry logic
    target = None

    if z > entry_threshold:
        # Short A, Long B (B size includes hedge ratio sign)
        if col == 0:
            target = -shares_a
        else:
            target = shares_b

    elif z < -entry_threshold:
        # Long A, Short B
        if col == 0:
            target = shares_a
        else:
            target = -shares_b

    else:
        # No entry/exit signal
        return (np.nan, 0, 0)

    # Compute order size as difference between target and current position
    order_size = float(target - pos_now)

    # If order size is effectively zero, do nothing
    if abs(order_size) < 1e-12:
        return (np.nan, 0, 0)

    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio (rolling OLS) and z-score of the spread between two assets.

    Accepts either pandas DataFrames/Series with a 'close' column or raw numpy
    arrays of close prices. Returns a dictionary with numpy arrays for:
      - close_a
      - close_b
      - hedge_ratio
      - zscore

    Notes:
    - Rolling OLS: for each index i >= hedge_lookback, regress Price_A on
      Price_B using the prior `hedge_lookback` observations (i-hedge_lookback:i).
    - Spread = Price_A - hedge_ratio * Price_B
    - Z-score is computed using a rolling mean/std over `zscore_lookback`.

    Edge cases:
    - If there are NaNs in the regression window, that hedge_ratio entry is left NaN.
    - Rolling z-score uses min_periods = zscore_lookback to ensure full-window stats.

    Returns:
        Dict[str, np.ndarray]
    """

    # Helper to extract close numpy array from input
    def _extract_close(x):
        if isinstance(x, (np.ndarray, list)):
            arr = np.asarray(x, dtype=float)
            if arr.ndim != 1:
                raise ValueError("Input price arrays must be one-dimensional")
            return arr
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x["close"].astype(float).values
        raise TypeError("asset_a/asset_b must be numpy array, pandas Series or DataFrame")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (ordinary least squares) using scipy.stats.linregress
    # Window uses values [i - hedge_lookback, i) - i.e., previous hedge_lookback bars
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]

        # If there are NaNs in the window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # Ensure there is variation in x; linregress will return nan slope otherwise
        if np.allclose(x, x[0]):
            hedge_ratio[i] = np.nan
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Z-score: (spread - mean) / std
    zscore = (spread_series - spread_mean) / spread_std

    # Convert to numpy arrays
    result = {
        "close_a": np.asarray(close_a, dtype=float),
        "close_b": np.asarray(close_b, dtype=float),
        "hedge_ratio": np.asarray(hedge_ratio, dtype=float),
        "zscore": np.asarray(zscore.values, dtype=float),
    }

    return result
