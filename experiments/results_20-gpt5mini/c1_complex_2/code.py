import numpy as np
import pandas as pd
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

    Notes:
        - If returning (np.nan, 0, 0), it signals no order for this asset at this bar.
        - The function uses fixed-dollar sizing per leg. Hedge ratio is used to
          scale the B leg's number of shares (magnitude).
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    if i < 0:
        return (np.nan, 0, 0)

    # Protect against out-of-bounds
    if i >= len(zscore) or i >= len(hedge_ratio) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not ready, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute shares based on fixed notional per leg
    shares_a = float(notional_per_leg / price_a)
    # Use magnitude of hedge ratio to compute B-side share magnitude; direction decided by signal
    shares_b = float(notional_per_leg / price_b * abs(hr))

    # Determine previous zscore for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Exit conditions
    crossing_exit = False
    if not np.isnan(prev_z):
        # Check crossing the exit_threshold (e.g., 0.0)
        if (prev_z - exit_threshold) * (z - exit_threshold) < 0:
            crossing_exit = True

    stop_loss = abs(z) > stop_threshold

    # If exit or stop-loss triggered, close the position for this asset
    if crossing_exit or stop_loss:
        # Close entire position
        if pos == 0.0:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Entry logic
    if z > entry_threshold:
        # Short Asset A, Long Asset B
        if col == 0:
            target = -shares_a
        else:
            target = +shares_b

    elif z < -entry_threshold:
        # Long Asset A, Short Asset B
        if col == 0:
            target = +shares_a
        else:
            target = -shares_b

    else:
        # No entry signal; do nothing
        return (np.nan, 0, 0)

    # Compute required delta to reach target (shares)
    delta = float(target - pos)

    # If no meaningful change, do nothing
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    # Return order: size in shares (Amount)
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
        asset_a: DataFrame with 'close' column for Asset A or a numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B or a numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept both DataFrame inputs (with 'close') and numpy arrays / Series
    if isinstance(asset_a, (np.ndarray, pd.Series)):
        close_a = np.asarray(asset_a, dtype=float).copy()
    elif isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise KeyError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].astype(float).values.copy()
    else:
        raise TypeError("asset_a must be a pandas.DataFrame with 'close' or a numpy array")

    if isinstance(asset_b, (np.ndarray, pd.Series)):
        close_b = np.asarray(asset_b, dtype=float).copy()
    elif isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise KeyError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].astype(float).values.copy()
    else:
        raise TypeError("asset_b must be a pandas.DataFrame with 'close' or a numpy array")

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio (slope of regression of A ~ B)
    # We require at least hedge_lookback points
    if hedge_lookback < 1:
        raise ValueError("hedge_lookback must be >= 1")

    for i in range(hedge_lookback, n + 1):
        # i is the end index for the window
        start = i - hedge_lookback
        end = i
        window_x = close_b[start:end]
        window_y = close_a[start:end]

        # Skip if any NaNs in the window or constant B (no variation)
        if np.isnan(window_x).any() or np.isnan(window_y).any():
            # Leave hedge_ratio[end-1] as NaN
            continue

        if np.allclose(window_x, window_x[0]):
            # No variation in x -> cannot regress
            continue

        # Compute slope using scipy.stats.linregress
        try:
            slope, intercept, r_value, p_value, stderr = stats.linregress(window_x, window_y)
            hedge_ratio[end - 1] = float(slope)
        except Exception:
            # In case regression fails for numerical reasons
            hedge_ratio[end - 1] = np.nan

    # Compute spread: close_a - hedge_ratio * close_b
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std for spread to compute z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
