import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


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

    # Defensive checks
    if i < 0:
        return (np.nan, 0, 0)

    # Read current zscore and hedge ratio
    z = zscore[i] if i < len(zscore) else np.nan
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If z or hedge ratio is not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Read prices
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine shares based on fixed notional per leg
    # Following the example: shares_a = notional / price_a
    # shares_b = notional / price_b * hedge_ratio
    shares_a = notional_per_leg / price_a
    shares_b = (notional_per_leg / price_b) * hr

    # Previous z for crossing detection
    z_prev = zscore[i - 1] if i - 1 >= 0 else np.nan

    # Stop-loss: absolute z beyond stop_threshold -> close positions
    if np.isfinite(z) and abs(z) > stop_threshold:
        # If there is a position, close it
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on z-score crossing zero
    if not np.isnan(z_prev) and z_prev * z < 0:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic
    # If z > entry_threshold: short A, long B
    # If z < -entry_threshold: long A, short B
    if z > entry_threshold:
        # Targets:
        target_a = -shares_a
        target_b = shares_b
    elif z < -entry_threshold:
        target_a = shares_a
        target_b = -shares_b
    else:
        # No new entries, hold current positions
        return (np.nan, 0, 0)

    # Return order for the current column based on target - current position
    target = target_a if col == 0 else target_b
    size = target - pos

    # If size is effectively zero, do nothing
    if np.isclose(size, 0.0):
        return (np.nan, 0, 0)

    # Use Amount (shares) ordering
    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array of closes)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array of closes)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame/Series with 'close' or raw numpy arrays
    # Extract close arrays
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.DataFrame):
            if 'close' not in asset_a.columns:
                raise KeyError("asset_a DataFrame must contain 'close' column")
            close_a = asset_a['close'].values.astype(float)
        else:
            close_a = asset_a.values.astype(float)
    elif isinstance(asset_a, (np.ndarray, list, tuple)):
        close_a = np.asarray(asset_a, dtype=float)
    else:
        raise TypeError('asset_a must be a pandas DataFrame/Series or numpy array')

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.DataFrame):
            if 'close' not in asset_b.columns:
                raise KeyError("asset_b DataFrame must contain 'close' column")
            close_b = asset_b['close'].values.astype(float)
        else:
            close_b = asset_b.values.astype(float)
    elif isinstance(asset_b, (np.ndarray, list, tuple)):
        close_b = np.asarray(asset_b, dtype=float)
    else:
        raise TypeError('asset_b must be a pandas DataFrame/Series or numpy array')

    if len(close_a) != len(close_b):
        raise ValueError('close arrays must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression to compute hedge ratio (slope) for each window
    # Use stats.linregress for simplicity
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Avoid division by zero: where std is 0 or nan, set zscore to nan
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        zscore = np.where(np.isfinite(zscore), zscore, np.nan)

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }
