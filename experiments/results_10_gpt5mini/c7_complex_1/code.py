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
    notional_per_leg: float,
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float)
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
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Sanity checks for prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine shares sizing. Size Asset A by fixed notional and scale Asset B by hedge ratio
    shares_a = float(notional_per_leg) / price_a
    shares_b = shares_a * hr

    # Stop-loss: if |z| > stop_threshold -> close any open position
    if abs(z) > stop_threshold:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on mean reversion: z-score crosses exit_threshold
    if i > 0:
        prev_z = zscore[i - 1]
        if not np.isnan(prev_z):
            crossed = False
            if (prev_z > exit_threshold) and (z <= exit_threshold):
                crossed = True
            if (prev_z < exit_threshold) and (z >= exit_threshold):
                crossed = True

            if crossed:
                if pos != 0.0:
                    return (-pos, 0, 0)
                return (np.nan, 0, 0)

    # Entry logic
    # When z > entry_threshold: Short Asset A, Long Asset B
    if z > entry_threshold:
        if col == 0:
            # Asset A: short
            if pos == 0.0:
                return (-shares_a, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: long
            if pos == 0.0:
                return (shares_b, 0, 0)
            return (np.nan, 0, 0)

    # When z < -entry_threshold: Long Asset A, Short Asset B
    if z < -entry_threshold:
        if col == 0:
            # Asset A: long
            if pos == 0.0:
                return (shares_a, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: short
            if pos == 0.0:
                return (-shares_b, 0, 0)
            return (np.nan, 0, 0)

    # No action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR a 1D numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """

    def _extract_close(x):
        # Accept DataFrame with 'close', Series, or ndarray
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            arr = np.asarray(x)

        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Close price input must be 1-dimensional')
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Prepare hedge ratio using rolling OLS (no lookahead): use up to hedge_lookback most recent points including current
    hedge_ratio = np.full(n, np.nan)
    if hedge_lookback < 1:
        raise ValueError('hedge_lookback must be >= 1')

    for i in range(n):
        # Use available history up to and including i, capped by hedge_lookback
        L = min(i + 1, hedge_lookback)
        if L < 2:
            continue
        start = i - L + 1
        y = close_a[start: i + 1]
        x = close_b[start: i + 1]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        if np.all(x == x[0]):
            # Cannot regress on constant x
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread where hedge ratio available
    spread = np.full(n, np.nan)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score (right-aligned)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    zscore = np.full(n, np.nan)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
