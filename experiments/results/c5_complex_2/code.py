import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_ratio: np.ndarray,
    zscore: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0
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
        hedge_ratio: Rolling hedge ratio array
        zscore: Z-score of spread array
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
    pos_now = float(getattr(c, "position_now", 0.0))

    # Defensive checks
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio)) else np.nan
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If any critical value is NaN, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute number of shares for each leg based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    shares_b = (notional_per_leg / price_b) * hr

    # Previous z for exit-on-cross detection
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Default: no action
    target_a = None
    target_b = None

    # Stop-loss: if |z| > stop_threshold, close positions
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0

    # Exit on mean reversion crossing zero
    elif (not np.isnan(z_prev)) and (np.sign(z_prev) != np.sign(z)):
        target_a = 0.0
        target_b = 0.0

    # Entry: z-score breaches thresholds
    elif z > entry_threshold:
        # Short Asset A, Long Asset B (scaled by hedge ratio)
        target_a = -shares_a
        target_b = +shares_b

    elif z < -entry_threshold:
        # Long Asset A, Short Asset B (scaled by hedge ratio)
        target_a = +shares_a
        target_b = -shares_b

    else:
        return (np.nan, 0, 0)

    # Determine which asset we're processing and what order to place
    if col == 0:
        # Asset A
        if target_a is None:
            return (np.nan, 0, 0)
        delta = float(target_a - pos_now)
        if abs(delta) < 1e-9:
            return (np.nan, 0, 0)
        return (delta, 0, 0)
    else:
        # Asset B
        if target_b is None:
            return (np.nan, 0, 0)
        delta = float(target_b - pos_now)
        if abs(delta) < 1e-9:
            return (np.nan, 0, 0)
        return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR 1D numpy array/Series of closes OR combined DataFrame with two columns
        asset_b: DataFrame with 'close' column for Asset B OR 1D numpy array/Series of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close series from various inputs
    def _extract_close(obj):
        if isinstance(obj, np.ndarray):
            return np.asarray(obj, dtype=float)
        if isinstance(obj, pd.Series):
            return obj.values.astype(float)
        if isinstance(obj, pd.DataFrame):
            # Prefer 'close' column if present
            if 'close' in obj.columns:
                return obj['close'].values.astype(float)
            # Otherwise, take the first column as close
            if obj.shape[1] >= 1:
                return obj.iloc[:, 0].values.astype(float)
        # Fallback: try to coerce
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass
        raise ValueError("Unable to extract close prices from provided object")

    # Attempt to extract close arrays
    close_a = None
    close_b = None
    try:
        close_a = _extract_close(asset_a)
        close_b = _extract_close(asset_b)
    except Exception:
        # asset_a might be a combined DataFrame with two asset columns
        if isinstance(asset_a, pd.DataFrame):
            df = asset_a
            if set(['asset_a', 'asset_b']).issubset(df.columns):
                close_a = df['asset_a'].values.astype(float)
                close_b = df['asset_b'].values.astype(float)
            elif df.shape[1] >= 2:
                close_a = df.iloc[:, 0].values.astype(float)
                close_b = df.iloc[:, 1].values.astype(float)
            else:
                raise
        else:
            raise

    # Align lengths: use the minimum length to avoid mismatched inputs (truncation ensures no lookahead)
    len_a = len(close_a)
    len_b = len(close_b)
    if len_a != len_b:
        n = min(len_a, len_b)
        close_a = np.asarray(close_a[:n], dtype=float)
        close_b = np.asarray(close_b[:n], dtype=float)
    else:
        n = len_a

    if n == 0:
        return {
            'close_a': np.array([], dtype=float),
            'close_b': np.array([], dtype=float),
            'hedge_ratio': np.array([], dtype=float),
            'zscore': np.array([], dtype=float),
        }

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Minimum OLS window to start computing hedge ratio
    min_ols = 20
    if hedge_lookback < min_ols:
        min_ols = hedge_lookback
    min_ols = max(2, min(min_ols, n))

    # Rolling OLS slope (hedge ratio). Causal: slope at time i uses data up to i-1 (window end exclusive)
    for i in range(min_ols, n):
        start = max(0, i - hedge_lookback)
        x = close_b[start:i]
        y = close_a[start:i]
        if len(x) < 2 or np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (causal: window includes current spread)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    zscore = (spread - spread_mean) / spread_std
    zscore = np.where(np.isfinite(zscore), zscore, np.nan)

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': hedge_ratio.astype(float),
        'zscore': zscore.astype(float),
    }
