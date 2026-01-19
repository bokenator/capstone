import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Union, Dict, Tuple


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func (flexible mode).

    Args:
        c: vectorbt OrderContext-like object with attributes:
           - c.i: current bar index (int)
           - c.col: asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float) [optional]
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array of spread z-scores (aligned with closes)
        hedge_ratio: 1D array of rolling hedge ratios (aligned with closes)
        entry_threshold: z-score threshold to enter (e.g., 2.0)
        exit_threshold: z-score threshold to exit (e.g., 0.0)
        stop_threshold: z-score threshold for stop-loss (e.g., 3.0)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        Tuple of (size, size_type, direction):
        - size: float (number of shares to buy/sell; positive=buy, negative=sell)
        - size_type: int (0=Amount, 1=Value, 2=Percent)
        - direction: int (0=Both, 1=LongOnly, 2=ShortOnly)

    Notes:
        - This implementation follows the strategy described in the prompt.
        - Uses simple fixed-notional sizing as: shares_a = notional_per_leg / price_a
          and shares_b = (notional_per_leg / price_b) * hedge_ratio.
        - Returns (np.nan, 0, 0) when no action is required.
    """
    # Extract context
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validation of indices
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If insufficient data, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Guard against invalid prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine share sizes based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    # Maintain hedge_ratio sign in sizing for Asset B
    shares_b = (notional_per_leg / price_b) * hr

    # Previous z for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine signals
    cross_zero = False
    if not np.isnan(prev_z):
        # Crossing the exit threshold (commonly 0.0)
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            cross_zero = True

    stop_loss = abs(z) > stop_threshold
    enter_short = z > entry_threshold  # Short Asset A, Long Asset B
    enter_long = z < -entry_threshold  # Long Asset A, Short Asset B

    # Compute targets (number of shares) for both assets
    target_a = 0.0
    target_b = 0.0

    if stop_loss or cross_zero:
        # Exit: set both targets to zero
        target_a = 0.0
        target_b = 0.0
    elif enter_short:
        # Short Asset A, Long Asset B (use hedge_ratio sign for B)
        target_a = -shares_a
        target_b = shares_b
    elif enter_long:
        # Long Asset A, Short Asset B
        target_a = shares_a
        target_b = -shares_b
    else:
        # No action
        return (np.nan, 0, 0)

    # Choose target for the current asset (col)
    target = target_a if col == 0 else target_b

    # Compute delta (how many shares to trade)
    size = float(target - pos_now)

    # If no change required, do nothing
    if np.isclose(size, 0.0):
        return (np.nan, 0, 0)

    # Return amount-based order (number of shares)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs trading strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B or 1D numpy array of closes
        hedge_lookback: lookback window for rolling OLS hedge ratio
        zscore_lookback: lookback window for z-score mean/std

    Returns:
        Dict with keys:
            'close_a' -> np.ndarray of closes for Asset A
            'close_b' -> np.ndarray of closes for Asset B
            'hedge_ratio' -> np.ndarray of rolling hedge slopes (NaN where unavailable)
            'zscore' -> np.ndarray of spread z-scores (NaN where unavailable)
    """
    # Extract close series depending on input types
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = np.asarray(asset_a['close'], dtype=float)
    elif isinstance(asset_a, np.ndarray):
        close_a = np.asarray(asset_a, dtype=float)
    else:
        raise TypeError("asset_a must be a pd.DataFrame or np.ndarray")

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = np.asarray(asset_b['close'], dtype=float)
    elif isinstance(asset_b, np.ndarray):
        close_b = np.asarray(asset_b, dtype=float)
    else:
        raise TypeError("asset_b must be a pd.DataFrame or np.ndarray")

    # Align lengths (use the minimum to be safe)
    n = min(len(close_a), len(close_b))
    if len(close_a) != n:
        close_a = close_a[:n]
    if len(close_b) != n:
        close_b = close_b[:n]

    # Initialize outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS to compute hedge ratio (slope of regression of A on B)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # If any NaNs in window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue
        # If x is constant, slope is undefined
        if np.allclose(x, x[0]):
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio (element-wise)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # z-score (guard against zero std)
    zscore = np.full(n, np.nan, dtype=float)
    denom = np.asarray(spread_std, dtype=float)
    mean_arr = np.asarray(spread_mean, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(denom)) & (denom > 0)
    zscore[valid] = (spread[valid] - mean_arr[valid]) / denom[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
