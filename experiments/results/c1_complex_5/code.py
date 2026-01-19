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
    notional_per_leg: float = 10000.0
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    Notes:
    - This function is designed to work with the runner's wrapper which calls
      the user function with positional args: (c, close_a, close_b, zscore, hedge_ratio, ...)
    - Returns (size, size_type, direction) where size is in shares (Amount, 0).

    Args:
        c: OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: Close price array for Asset A
        close_b: Close price array for Asset B
        zscore: Z-score array of the spread
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score entry threshold
        exit_threshold: Z-score exit threshold (usually 0.0)
        stop_threshold: Z-score stop-loss threshold
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        Tuple (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic sanity checks
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Extract current indicators and prices
    z = float(zscore[i]) if i < len(zscore) else np.nan
    h = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If z-score is not available, do nothing
    if np.isnan(z) or np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute previous z for exit crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and i - 1 < len(zscore) else np.nan

    # Compute target positions (in shares) for each asset
    # Default: no new target (keep existing)
    target_a = None
    target_b = None

    # Stop-loss takes precedence
    if not np.isnan(z) and abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0

    # Exit if z-score crosses the exit_threshold (e.g., 0.0)
    elif not np.isnan(prev_z):
        crossed_down = (prev_z > exit_threshold and z <= exit_threshold)
        crossed_up = (prev_z < exit_threshold and z >= exit_threshold)
        if crossed_down or crossed_up:
            target_a = 0.0
            target_b = 0.0

    # Entry conditions
    if target_a is None and target_b is None:
        # Need a valid hedge ratio to open new pair trades
        if np.isnan(h):
            # Cannot construct hedge without ratio
            return (np.nan, 0, 0)

        # Determine number of shares based on fixed notional per leg
        shares_a = notional_per_leg / price_a
        shares_b = notional_per_leg / price_b * h

        if z > entry_threshold:
            # Short Asset A, Long Asset B
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Long Asset A, Short Asset B
            target_a = +shares_a
            target_b = -shares_b
        else:
            # No entry/exit signal
            return (np.nan, 0, 0)

    # Determine which asset we're producing an order for
    if col == 0:
        # Asset A
        if target_a is None:
            return (np.nan, 0, 0)
        delta = target_a - pos
    else:
        # Asset B
        if target_b is None:
            return (np.nan, 0, 0)
        delta = target_b - pos

    # If delta is effectively zero, do nothing
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    # Return order in Amount (number of shares)
    # size_type = 0 -> Amount (shares), direction = 0 -> Both
    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with a 'close' column or raw numpy arrays.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Accept numpy arrays or DataFrames
    if isinstance(asset_a, (np.ndarray, list)):
        close_a = np.asarray(asset_a, dtype=float)
    elif isinstance(asset_a, pd.DataFrame) or isinstance(asset_a, pd.Series):
        # DataFrame: expect a 'close' column
        if isinstance(asset_a, pd.Series):
            close_a = asset_a.values.astype(float)
        else:
            if 'close' not in asset_a.columns:
                raise KeyError("asset_a DataFrame must contain 'close' column")
            close_a = asset_a['close'].values.astype(float)
    else:
        raise TypeError('asset_a must be a numpy array or pandas DataFrame/Series')

    if isinstance(asset_b, (np.ndarray, list)):
        close_b = np.asarray(asset_b, dtype=float)
    elif isinstance(asset_b, pd.DataFrame) or isinstance(asset_b, pd.Series):
        if isinstance(asset_b, pd.Series):
            close_b = asset_b.values.astype(float)
        else:
            if 'close' not in asset_b.columns:
                raise KeyError("asset_b DataFrame must contain 'close' column")
            close_b = asset_b['close'].values.astype(float)
    else:
        raise TypeError('asset_b must be a numpy array or pandas DataFrame/Series')

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Compute rolling hedge ratio (OLS slope of y=close_a on x=close_b)
    hedge_ratio = np.full(n, np.nan)

    # Ensure lookbacks are sensible
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        mask = ~np.isnan(y) & ~np.isnan(x)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = np.full(n, np.nan)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
    zscore[~np.isfinite(zscore)] = np.nan

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
