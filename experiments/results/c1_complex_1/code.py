import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple, Union


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
    Generate orders for pairs trading. Called by vectorbt's from_order_func wrapper.

    Args:
        c: Order context with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array with z-score values for the spread
        hedge_ratio: 1D array with rolling hedge ratio (beta)
        entry_threshold: threshold to enter (positive side)
        exit_threshold: threshold to exit (usually 0.0)
        stop_threshold: stop-loss threshold (absolute z-score)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
          size: number of shares (positive=buy, negative=sell)
          size_type: 0=Amount (shares), 1=Value ($), 2=Percent
          direction: 0=Both, 1=LongOnly, 2=ShortOnly

    Notes:
        - Uses amount-based orders (size_type=0) with fractional shares allowed.
        - This function is designed to be called in flexible multi-asset mode where the
          wrapper calls it once per asset and sequentially executes returned orders.
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation of indices
    if i < 0:
        return (np.nan, 0, 0)

    # Guard against out-of-bounds or missing data
    try:
        price_a = float(close_a[i])
        price_b = float(close_b[i])
        z = float(zscore[i])
        beta = float(hedge_ratio[i])
    except Exception:
        return (np.nan, 0, 0)

    # If any critical value is NaN, do nothing
    if not np.isfinite(price_a) or not np.isfinite(price_b) or not np.isfinite(z) or not np.isfinite(beta):
        return (np.nan, 0, 0)

    # Determine number of shares based on fixed notional per leg
    # shares_a: base shares for Asset A
    # shares_b: scaled by hedge ratio (units of B per base notional)
    # Follow the prompt example: shares_b = (notional / price_b) * hedge_ratio
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * beta)

    # Exit conditions
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    cross_to_exit = False
    if i > 0 and np.isfinite(prev_z) and np.isfinite(z):
        # Detect crossing of zscore through exit_threshold (commonly 0.0)
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            cross_to_exit = True

    stop_loss = abs(z) > stop_threshold

    # If we need to close (either crossing or stop-loss) and we have a position, close it
    if (cross_to_exit or stop_loss) and pos != 0.0:
        # Close entire position for this asset
        return (-pos, 0, 0)

    # Entry logic only when there's no existing position in this asset
    if pos == 0.0:
        # Short A, Long B when zscore > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B (hedge_ratio units scaled by notional)
                return (shares_b, 0, 0)

        # Long A, Short B when zscore < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B
                return (-shares_b, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for a pairs trading strategy.

    Accepts either DataFrames with a 'close' column or plain 1D numpy arrays of close prices.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or 1D numpy array
        asset_b: DataFrame with 'close' column for Asset B or 1D numpy array
        hedge_lookback: lookback period for rolling OLS regression to compute hedge ratio
        zscore_lookback: lookback period for rolling mean/std for z-score

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Extract close arrays from inputs
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].astype(float).to_numpy()
    else:
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].astype(float).to_numpy()
    else:
        close_b = np.asarray(asset_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close arrays must be 1-dimensional")

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Initialize hedge ratio with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (using previous hedge_lookback points, excluding current index)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Require full window and finite values
        if x.size != hedge_lookback or y.size != hedge_lookback:
            continue
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue
        if np.std(x) == 0.0:
            # Cannot regress if x has zero variance
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread: spread = A - beta * B
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().to_numpy()

    # z-score
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
