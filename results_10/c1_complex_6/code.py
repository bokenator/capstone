import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Tuple, Dict


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This function is implemented for flexible=True (multi-asset). It will be
    called once per asset per bar (simulated by the wrapper in the backtest
    runner). It returns a tuple (size, size_type, direction).

    Parameters
    ----------
    c : OrderContext-like object
        Attributes used: c.i (int), c.col (int: 0=A,1=B), c.position_now (float), c.cash_now (float)
    close_a, close_b : np.ndarray
        Close price arrays for Asset A and Asset B
    zscore : np.ndarray
        Z-score array for the spread
    hedge_ratio : np.ndarray
        Rolling hedge ratio (slope) array
    entry_threshold : float
        Threshold for entering trades (e.g., 2.0)
    exit_threshold : float
        Threshold for exiting trades (e.g., 0.0)
    stop_threshold : float
        Threshold for stop-loss (e.g., 3.0)
    notional_per_leg : float
        Fixed notional per leg in dollars (e.g., 10000.0)

    Returns
    -------
    tuple
        (size, size_type, direction) as described in the prompt.
    """

    # Extract context
    i: int = int(getattr(c, "i", 0))
    col: int = int(getattr(c, "col", 0))

    # Current position for this asset (shares)
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos = float(pos_now)
    except Exception:
        pos = 0.0

    # Basic bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Current indicators/prices
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # If key data is missing, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Prevent division by zero or nonsensical sizes
    if price_a <= 0 or price_b <= 0 or notional_per_leg <= 0:
        return (np.nan, 0, 0)

    # Determine share sizes based on fixed notional per leg.
    # shares_a: number of Asset A shares corresponding to notional_per_leg
    # shares_b: number of Asset B shares scaled by hedge ratio. This follows
    # the example in the prompt where shares_b = (notional / price_b) * hedge_ratio.
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Stop-loss: absolute zscore beyond stop_threshold -> close positions
    if abs(z) > stop_threshold:
        # Close whatever position exists on this asset
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Exit on crossing zero (use previous zscore to detect crossing)
    crossed_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        # If previous was positive and current is <= exit_threshold -> crossed downwards
        if prev_z > 0 and z <= exit_threshold:
            crossed_zero = True
        # If previous was negative and current is >= exit_threshold -> crossed upwards
        if prev_z < 0 and z >= exit_threshold:
            crossed_zero = True

    if crossed_zero:
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Entry logic
    # z > entry_threshold  => SHORT A, LONG B (hedge_ratio units)
    # z < -entry_threshold => LONG A, SHORT B (hedge_ratio units)
    if z > entry_threshold:
        if col == 0:
            # Asset A should be short: desired position = -shares_a
            desired = -shares_a
            delta = desired - pos
            if abs(delta) < 1e-12:
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)
        else:
            # Asset B should be long: desired position = +shares_b
            desired = shares_b
            delta = desired - pos
            if abs(delta) < 1e-12:
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)

    if z < -entry_threshold:
        if col == 0:
            # Asset A should be long: desired position = +shares_a
            desired = shares_a
            delta = desired - pos
            if abs(delta) < 1e-12:
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)
        else:
            # Asset B should be short: desired position = -shares_b
            desired = -shares_b
            delta = desired - pos
            if abs(delta) < 1e-12:
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible and accepts either:
      - pandas DataFrames with a 'close' column (as per DATA_SCHEMA), or
      - numpy arrays / pandas Series of close prices.

    Returns a dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    All values are numpy arrays of the same length.
    """

    # Accept either DataFrame/Series or raw numpy arrays
    def _extract_close(x):
        # DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return x["close"].astype(float).values
        # Series
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # Numpy array-like
        arr = np.asarray(x)
        if arr.ndim == 0:
            raise ValueError("close series must be 1-dimensional")
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Validate lookbacks
    if not isinstance(hedge_lookback, int) or hedge_lookback < 1:
        raise ValueError("hedge_lookback must be a positive integer")
    if not isinstance(zscore_lookback, int) or zscore_lookback < 1:
        raise ValueError("zscore_lookback must be a positive integer")

    if hedge_lookback > n:
        raise ValueError("hedge_lookback cannot be larger than the number of observations")
    if zscore_lookback > n:
        # Allow but will produce all NaNs
        pass

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (slope) to compute hedge ratio
    # For each window ending at index i (exclusive of i), use values [i-hedge_lookback:i]
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]

        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x has zero variance, slope is undefined
        if np.isclose(x.std(ddof=0), 0.0):
            hedge_ratio[i] = np.nan
            continue

        # Ordinary least squares slope
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread and rolling z-score
    spread = close_a - hedge_ratio * close_b

    # Use pandas rolling to compute mean and std (require full window to produce values)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use population std (ddof=0) to avoid NaNs when lookback == 1 (though schema min is 5)
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        "close_a": np.asarray(close_a, dtype=float),
        "close_b": np.asarray(close_b, dtype=float),
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }