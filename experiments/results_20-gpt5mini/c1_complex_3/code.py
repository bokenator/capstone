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
    notional_per_leg: float
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This function uses flexible=True (multi-asset). It returns a tuple
    (size, size_type, direction) where size_type: 0=Amount (shares), 1=Value ($), 2=Percent.

    Args:
        c: Order context provided by vectorbt wrapper. Expected attributes:
           - c.i (int): current bar index
           - c.col (int): asset column (0=Asset A, 1=Asset B)
           - c.position_now (float): current position for this asset (in shares)
           - c.cash_now (float): current cash (not used here)
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array of z-score values for the spread
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: Z-score threshold to enter (e.g., 2.0)
        exit_threshold: Z-score threshold to exit (e.g., 0.0)
        stop_threshold: Z-score threshold for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional amount per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Basic sanity checks
    if i < 0:
        return (np.nan, 0, 0)

    # Guard array bounds
    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If either price is invalid, do nothing
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    hedge = float(hedge_ratio[i])
    if not np.isfinite(hedge):
        # Hedge ratio unknown -> avoid trading
        return (np.nan, 0, 0)

    # Number of shares for Asset A based on fixed notional
    shares_a = notional_per_leg / price_a
    # Number of shares for Asset B determined by hedge ratio relative to shares_a
    shares_b = hedge * shares_a

    # Previous z for exit-crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Stop-loss has highest priority: if breached, close current position for this asset
    if abs(z) > stop_threshold:
        if pos != 0.0:
            # Close full position for this asset
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit condition: z crosses exit_threshold (default 0.0)
    crossed_exit = False
    if np.isfinite(prev_z):
        # Crossing logic for threshold (handles both directions)
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed_exit = True

    if crossed_exit:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic
    # If z > entry_threshold -> short A, long B
    # If z < -entry_threshold -> long A, short B
    target_a = 0.0
    target_b = 0.0

    if z > entry_threshold:
        target_a = -shares_a
        target_b = +shares_b
    elif z < -entry_threshold:
        target_a = +shares_a
        target_b = -shares_b
    else:
        # No entry/exit signal -> do nothing
        return (np.nan, 0, 0)

    # Determine which asset we're processing and return the required order delta
    if col == 0:
        # Asset A
        delta = target_a - pos
        # If no change required, do nothing
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)
    else:
        # Asset B
        delta = target_b - pos
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with a 'close' column or 1D numpy arrays / pandas Series.

    Args:
        asset_a: DataFrame/Series/ndarray for Asset A close prices (if DataFrame, must contain 'close')
        asset_b: DataFrame/Series/ndarray for Asset B close prices (if DataFrame, must contain 'close')
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' mapping to numpy arrays
    """
    # Helper to extract close price array from input
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            # Try to coerce
            arr = np.asarray(x)
        # Ensure 1-D
        arr = np.asarray(arr).astype(float)
        if arr.ndim != 1:
            raise ValueError('Close price input must be one-dimensional')
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    # Rolling hedge ratio via OLS (A ~ beta * B)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    for t in range(hedge_lookback, n + 1):
        i = t - 1  # index of the last element in the window
        start = t - hedge_lookback
        y = close_a[start:t]
        x = close_b[start:t]
        # Drop NaNs within the window
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Insufficient data to estimate
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using the (possibly NaN) hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Convert to numpy arrays and return
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }