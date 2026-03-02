import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Any


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

    This implementation uses fixed notional sizing for Asset A and scales Asset B
    by the rolling hedge ratio to preserve the unit relationship between the two
    assets (i.e., if you trade X shares of A, you trade hedge_ratio * X shares of B).

    Notes:
    - Uses amount (shares) sizing (size_type=0).
    - Returns (np.nan, 0, 0) when no action is required.

    Args and return format as specified in the prompt.
    """
    i = int(c.i)
    col = int(c.col)

    # Defensive checks
    if i < 0:
        return (np.nan, 0, 0)

    # Read current indicators/prices
    try:
        z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    # If z-score or hedge ratio is NaN, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    if np.isnan(hr):
        # Cannot size hedge without hedge ratio
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Invalid prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute base shares such that Asset A has fixed notional per leg
    # We set shares_a = notional_per_leg / price_a (number of shares of A)
    # and scale Asset B by the hedge ratio: shares_b = shares_a * hedge_ratio
    # This preserves the unit hedge relationship described in the specification.
    shares_a = float(notional_per_leg) / price_a
    shares_b = shares_a * hr

    # Determine previous z for crossing detection (use exit_threshold)
    prev_z = np.nan
    if i > 0:
        prev_z = zscore[i - 1]

    # Exit conditions
    crossed_exit_threshold = False
    if not np.isnan(prev_z):
        # Detect crossing of the configured exit_threshold
        try:
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed_exit_threshold = True
        except Exception:
            crossed_exit_threshold = False

    stop_loss = abs(z) > stop_threshold

    # Determine target positions (in shares) for both assets
    target_a = None
    target_b = None

    if crossed_exit_threshold or stop_loss:
        # Close both legs
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry logic
        if z > entry_threshold:
            # Spread wide (A expensive relative to B): SHORT A, LONG B
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Spread low (A cheap relative to B): LONG A, SHORT B
            target_a = +shares_a
            target_b = -shares_b
        else:
            # No new signals
            target_a = None
            target_b = None

    # Determine current position for this asset
    pos_now = c.position_now
    if pos_now is None or (isinstance(pos_now, float) and np.isnan(pos_now)):
        pos_now = 0.0

    # Select target depending on column
    if col == 0:
        tgt = target_a
    else:
        tgt = target_b

    # No action if no target decided
    if tgt is None:
        return (np.nan, 0, 0)

    # Compute order as difference between target and current position
    order_size = float(tgt) - float(pos_now)

    # If order is effectively zero, do nothing
    if np.isclose(order_size, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return number-of-shares order (size_type=0), allow both directions (direction=0)
    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs trading strategy.

    Accepts either DataFrames (with a 'close' column) or numpy arrays for asset inputs.

    Returns a dict with keys:
      - 'close_a': np.ndarray of close prices for asset A
      - 'close_b': np.ndarray of close prices for asset B
      - 'hedge_ratio': np.ndarray of rolling OLS slope (NaN where not available)
      - 'zscore': np.ndarray of z-scores of the spread (NaN where not available)
    """
    # Helper to extract close arrays
    def _get_close(series_or_array):
        if isinstance(series_or_array, pd.DataFrame):
            if 'close' not in series_or_array.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            arr = series_or_array['close'].values
        elif isinstance(series_or_array, pd.Series):
            arr = series_or_array.values
        elif isinstance(series_or_array, np.ndarray):
            arr = series_or_array
        else:
            # Try to coerce
            arr = np.asarray(series_or_array)
        if arr.ndim != 1:
            raise ValueError('Input close data must be a 1D array/Series')
        return arr.astype(float)

    close_a = _get_close(asset_a)
    close_b = _get_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Input arrays must have the same length')

    n = len(close_a)

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be at least 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be at least 1')

    # Rolling OLS to compute hedge ratio (slope of regression of A on B)
    # For each window end at i (inclusive of i-1), place result at index i
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # Mask NaNs within window
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            # Not enough data to compute regression
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread = A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use population std (ddof=0) to avoid NaNs when window full
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    zscore = np.full(n, np.nan, dtype=float)
    mask_valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std > 0)
    zscore[mask_valid] = (spread[mask_valid] - spread_mean[mask_valid]) / spread_std[mask_valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
