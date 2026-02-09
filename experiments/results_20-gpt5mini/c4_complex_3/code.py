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
        c: vectorbt OrderContext with attributes: i, col, position_now, cash_now
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array for the spread
        hedge_ratio: Rolling hedge ratio array (slope of regression)
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars (e.g., 10000.0)

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Validate index
    if i < 0:
        return (np.nan, 0, 0)

    # Bounds check for arrays
    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z = float(zscore[i])

    # If z-score is not available, do nothing (but allow closes using pos if needed)
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Basic price validation
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio)) else np.nan

    # Compute share counts based on fixed notional per leg
    # shares_a: number of A shares for notional_per_leg
    shares_a = notional_per_leg / price_a if price_a > 0 else np.nan
    # shares_b: scaled by hedge ratio to maintain hedge in units (as per prompt example)
    shares_b = (notional_per_leg / price_b * hr) if (price_b > 0 and not np.isnan(hr)) else np.nan

    # Helper to close a position for this asset
    def _close() -> tuple:
        if pos == 0:
            return (np.nan, 0, 0)
        # Return order to offset current position to zero
        return (-pos, 0, 0)

    # 1) Stop-loss: if |z| > stop_threshold -> close existing positions
    if stop_threshold is not None and not np.isnan(stop_threshold):
        if abs(z) > float(stop_threshold):
            return _close()

    # 2) Exit condition: crossing of exit_threshold
    # Detect crossing of threshold between previous and current bar
    crossed = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev = float(zscore[i - 1])
        t = float(exit_threshold)
        try:
            if (prev - t) * (z - t) < 0:
                crossed = True
        except Exception:
            crossed = False

    # Also consider exact equality to threshold as an exit
    if z == float(exit_threshold):
        crossed = True

    if crossed:
        return _close()

    # 3) Entry logic: only open if no current position for this asset
    if pos == 0:
        # Short A, Long B when z > entry_threshold
        if z > float(entry_threshold):
            if col == 0:
                # Asset A: short
                if np.isnan(shares_a) or shares_a == 0:
                    return (np.nan, 0, 0)
                return (-float(shares_a), 0, 0)
            else:
                # Asset B: long hedge-sized position
                if np.isnan(shares_b) or shares_b == 0:
                    return (np.nan, 0, 0)
                return (float(shares_b), 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -float(entry_threshold):
            if col == 0:
                # Asset A: long
                if np.isnan(shares_a) or shares_a == 0:
                    return (np.nan, 0, 0)
                return (float(shares_a), 0, 0)
            else:
                # Asset B: short hedge-sized position
                if np.isnan(shares_b) or shares_b == 0:
                    return (np.nan, 0, 0)
                return (-float(shares_b), 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR 1D numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """

    def _to_close_array(x):
        # Accept DataFrame with 'close', Series, or 1D numpy array
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame input must contain 'close' column")
            arr = x['close'].astype(float).values
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Allow list-like
            arr = np.asarray(x, dtype=float)

        # Ensure 1D
        arr = arr.reshape(-1)
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 1 or zscore_lookback < 1:
        raise ValueError('Lookback windows must be positive integers')

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Rolling hedge ratio using OLS with lookback on prior data (no lookahead)
    # hedge_ratio[i] is computed using data from [i - hedge_lookback, i)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Require full non-NaN window
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # Require non-zero variance in x
        if np.allclose(x, x[0]):
            hedge_ratio[i] = np.nan
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread using hedge_ratio estimated at each time
    for i in range(n):
        hr = hedge_ratio[i]
        if not np.isnan(hr) and not np.isnan(close_a[i]) and not np.isnan(close_b[i]):
            spread[i] = close_a[i] - hr * close_b[i]
        else:
            spread[i] = np.nan

    # Rolling mean and std for z-score (use min_periods = zscore_lookback to avoid small-sample z-scores)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score where possible
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }