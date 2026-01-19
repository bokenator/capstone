import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0
) -> tuple:
    """
    Generate orders for pairs trading. Designed to be called in flexible mode.

    Note: The wrapper used in the backtest calls this function with the
    arrays in the order (close_a, close_b, zscore, hedge_ratio, ...).

    Args:
        c: Order context with attributes i (index), col (0=A,1=B), position_now, cash_now
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array of z-score values (spread z-score)
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: threshold to enter (e.g., 2.0)
        exit_threshold: threshold to exit (e.g., 0.0) - used for crossing detection
        stop_threshold: threshold to force stop-loss (e.g., 3.0)
        notional_per_leg: fixed dollar amount per leg (default 10000.0)

    Returns:
        (size, size_type, direction) tuple understood by wrapper:
         - size: float (positive buy, negative sell), or np.nan for no action
         - size_type: 0=Amount (shares), 1=Value ($), 2=Percent
         - direction: 0=Both, 1=LongOnly, 2=ShortOnly
    """
    i = int(getattr(c, 'i', 0))
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic bounds check
    n = max(len(close_a), len(close_b), len(zscore), len(hedge_ratio))
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Safely index arrays (they may be numpy arrays)
    try:
        z = float(zscore[i])
    except Exception:
        z = np.nan
    try:
        hr = float(hedge_ratio[i])
    except Exception:
        hr = np.nan

    # If indicators unavailable, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Prices
    try:
        price_a = float(close_a[i])
    except Exception:
        price_a = np.nan
    try:
        price_b = float(close_b[i])
    except Exception:
        price_b = np.nan

    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute share counts based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    shares_b_unit = notional_per_leg / price_b

    # Helper: detect zero crossing of z-score (search previous non-nan value)
    crossed_zero = False
    if i > 0:
        j = i - 1
        z_prev = np.nan
        while j >= 0:
            try:
                z_prev = float(zscore[j])
            except Exception:
                z_prev = np.nan
            if not np.isnan(z_prev):
                break
            j -= 1
        if not np.isnan(z_prev) and (z_prev * z < 0):
            crossed_zero = True

    # Stop-loss: if |z| > stop_threshold, close any open position for this asset
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        if pos != 0.0:
            # Close entire position by returning negative of current position
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit condition: z-score crossed zero
    if crossed_zero:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry conditions (only when no current position)
    if pos == 0.0:
        # Short Asset A, Long Asset B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B in hedge_ratio units
                # units for B = hr * shares_b_unit
                size_b = hr * shares_b_unit
                return (size_b, 0, 0)

        # Long Asset A, Short Asset B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B in hedge_ratio units
                size_b = -hr * shares_b_unit
                return (size_b, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    This function accepts either DataFrames with a 'close' column or 1D numpy arrays.

    Args:
        asset_a: DataFrame or 1D array for Asset A (expects 'close' column if DataFrame)
        asset_b: DataFrame or 1D array for Asset B (expects 'close' column if DataFrame)
        hedge_lookback: lookback for rolling OLS to compute hedge ratio (int)
        zscore_lookback: lookback for rolling mean/std of spread (int)

    Returns:
        dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' (all numpy arrays)
    """
    # Extract close arrays from DataFrame or accept numpy arrays
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.Series):
                return x.values.astype(float)
            if 'close' in x.columns:
                return x['close'].values.astype(float)
            # fallback: if single-column DF, take the first column
            if x.shape[1] == 1:
                return x.iloc[:, 0].values.astype(float)
            raise KeyError("asset DataFrame must contain a 'close' column")
        # If numpy array or list-like
        arr = np.asarray(x)
        if arr.ndim == 0:
            raise ValueError("Close price input must be 1D array-like")
        if arr.ndim > 1:
            # If 2D with one column, flatten
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                # If given as 2D, but not recognized, try to flatten
                arr = arr.ravel()
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Validate lookbacks
    if hedge_lookback < 1:
        raise ValueError("hedge_lookback must be >= 1")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling OLS to compute hedge ratio (slope of regression of A on B)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Skip if any NaNs in the window
        if np.isnan(y).any() or np.isnan(x).any():
            hedge_ratio[i] = np.nan
            continue

        # Compute OLS slope (A ~ slope * B)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: spread = Price_A - hedge_ratio * Price_B
    spread = np.full(n, np.nan, dtype=float)
    valid_mask = ~np.isnan(hedge_ratio)
    spread[valid_mask] = close_a[valid_mask] - hedge_ratio[valid_mask] * close_b[valid_mask]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # z-score: (spread - mean) / std
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Ensure arrays are numpy float arrays
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }
