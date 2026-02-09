import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any


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
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If no valid zscore, skip
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices and hedge
    price_a = float(close_a[i])
    price_b = float(close_b[i])
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # Validate prices and hedge ratio
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)
    if not np.isfinite(hr) or hr == 0 or np.isnan(hr):
        # Without a valid hedge ratio we cannot size the B leg properly
        return (np.nan, 0, 0)

    # Compute share sizes based on notional per leg and hedge ratio
    # We size A to have notional_per_leg dollars, then scale B by hedge ratio
    shares_a = notional_per_leg / price_a
    shares_b = hr * shares_a

    # Helper to close position for this asset
    def close_pos() -> tuple:
        if pos == 0:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # 1) Stop-loss: if |z| > stop_threshold -> close positions
    if abs(z) > stop_threshold:
        return close_pos()

    # 2) Exit condition: z-score crosses zero (sign change)
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan
    crossed_zero = False
    if not np.isnan(prev_z):
        # crossing zero if signs differ and neither is zero
        crossed_zero = (prev_z * z) < 0

    if crossed_zero:
        return close_pos()

    # 3) Entry conditions
    # Short A, Long B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            # Asset A: short
            if pos == 0:
                return (-shares_a, 0, 0)
            # If current position is opposite (long), close it first
            if pos > 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: long (hedge_ratio units per A)
            if pos == 0:
                return (shares_b, 0, 0)
            if pos < 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Long A, Short B when z < -entry_threshold
    if z < -entry_threshold:
        if col == 0:
            # Asset A: long
            if pos == 0:
                return (shares_a, 0, 0)
            if pos < 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: short
            if pos == 0:
                return (-shares_b, 0, 0)
            if pos > 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array/series of closes)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array/series of closes)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close arrays from inputs that may be DataFrames, Series or numpy arrays
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            # Try to coerce to numpy array
            arr = np.asarray(x)
        # Ensure 1D
        arr = np.asarray(arr).astype(float)
        if arr.ndim != 1:
            raise ValueError('Close price input must be 1-dimensional')
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset A and Asset B must have the same number of observations')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be at least 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be at least 1')

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (slope only) with lookback window
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue
        # If constant series, linregress may still return slope=0; that's acceptable
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread and z-score
    spread = close_a - hedge_ratio * close_b

    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Avoid division by zero: where std is zero or NaN, set zscore to NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        zscore = (spread_s - spread_mean) / spread_std

    # Convert to numpy arrays
    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore.values.astype(float),
    }
