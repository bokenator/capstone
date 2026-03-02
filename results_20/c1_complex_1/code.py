import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Tuple, Dict


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
    Generate orders for a pairs trading strategy (flexible multi-asset mode).

    This function computes target positions for Asset A and Asset B based on
    the z-score of the spread and a rolling hedge ratio. It returns the
    incremental order size (in shares) required to move the current position
    to the target.

    Notes:
    - Uses base sizing of `notional_per_leg` for Asset A. Asset B is sized
      relative to Asset A using the rolling hedge ratio (units of B per unit A).
    - Returns (np.nan, 0, 0) when no order is required.

    Args:
        c: vectorbt OrderContext-like object with attributes:
           - c.i (int): current bar index
           - c.col (int): current asset column (0 = Asset A, 1 = Asset B)
           - c.position_now (float): current position size for the asset (shares)
           - c.cash_now (float): current cash balance (optional)
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array of z-score for the spread
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: z-score level to enter (positive number)
        exit_threshold: z-score level to exit (usually 0.0)
        stop_threshold: z-score level for stop-loss
        notional_per_leg: fixed notional amount per leg in dollars

    Returns:
        (size, size_type, direction) where size is number of shares to buy (>0)
        or sell (<0), size_type is 0 (Amount), and direction is 0 (Both).
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic bounds checks
    if i < 0:
        return (np.nan, 0, 0)

    # Safely extract current zscore and hedge ratio
    z = zscore[i] if i < len(zscore) else np.nan
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If z is not finite, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Prices for this bar
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        # Cannot size positions without valid prices
        return (np.nan, 0, 0)

    # Previous z (for crossing detection)
    prev_z = zscore[i - 1] if i > 0 and (i - 1) < len(zscore) else np.nan

    # Determine signals
    stop_loss = np.isfinite(z) and (abs(z) > stop_threshold)
    exit_cross = False
    if np.isfinite(prev_z):
        # Exit when z crosses the exit_threshold (e.g., 0.0)
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            exit_cross = True

    enter_short_spread = z > entry_threshold
    enter_long_spread = z < -entry_threshold

    # Compute base shares for Asset A using fixed notional per leg
    try:
        base_shares_a = float(notional_per_leg) / float(price_a)
    except Exception:
        return (np.nan, 0, 0)

    # Determine target positions (in shares) for both assets
    # Default: keep current position
    target_a = pos_now if col == 0 else None
    # We don't have cross-asset current position here; compute generic targets
    target_a_generic = None
    target_b_generic = None

    # Closing conditions take precedence
    if stop_loss or exit_cross:
        target_a_generic = 0.0
        target_b_generic = 0.0
    elif enter_short_spread or enter_long_spread:
        # Need a valid hedge ratio to size Asset B
        if not np.isfinite(hr):
            # Cannot size hedge leg without hedge ratio
            return (np.nan, 0, 0)

        if enter_short_spread:
            # Short spread: short A, long hr * A_units of B
            target_a_generic = -base_shares_a
            # B units per 1 A unit = hr -> scale by base_shares_a
            target_b_generic = hr * base_shares_a
        else:  # enter_long_spread
            # Long spread: long A, short hr * A_units of B
            target_a_generic = base_shares_a
            target_b_generic = -hr * base_shares_a
    else:
        # No signal -> no change
        return (np.nan, 0, 0)

    # Choose the correct target depending on which column we're evaluating
    if col == 0:
        # Asset A
        target = float(target_a_generic)
    else:
        # Asset B
        target = float(target_b_generic)

    # Compute order size as difference between desired target and current position
    size = target - pos_now

    # If size is effectively zero, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return order in Amount (shares), allow both long and short
    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and spread z-score for a pairs trading strategy.

    Accepts either DataFrames/Series with a 'close' column or plain 1D numpy arrays
    containing close prices. Returns a dictionary of numpy arrays:
    - 'close_a', 'close_b', 'hedge_ratio', 'zscore'

    Args:
        asset_a: DataFrame or Series with 'close' column (or a numpy array)
        asset_b: DataFrame or Series with 'close' column (or a numpy array)
        hedge_lookback: window size for rolling OLS hedge ratio
        zscore_lookback: window size for z-score rolling mean/std

    Returns:
        Dict[str, np.ndarray]
    """
    # Extract close price arrays from inputs (accept both DataFrame/Series and ndarray)
    def _extract_close(obj):
        # If it's a numpy array, assume it's already close prices
        if isinstance(obj, np.ndarray):
            return obj.astype(float)
        # If it's a pandas Series, use its values
        if isinstance(obj, pd.Series):
            return obj.values.astype(float)
        # If it's a DataFrame, only access the 'close' column as per DATA_SCHEMA
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise KeyError("DataFrame input must contain 'close' column")
            return obj['close'].astype(float).values
        # Fallback: try to convert
        return np.asarray(obj, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Input price arrays must have the same length')

    n = len(close_a)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio: slope of regression of A on B over lookback
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be at least 2')

    for end in range(hedge_lookback, n + 1):
        i = end - 1
        start = end - hedge_lookback
        y = close_a[start:end]
        x = close_b[start:end]

        # If any NaNs are present in the window, skip
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            hedge_ratio[i] = np.nan
            continue

        # If x has zero variance, skip to avoid division by zero
        if np.isclose(np.std(x), 0.0):
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using the hedge ratio (element-wise)
    # Where hedge_ratio is NaN, spread will be NaN
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Ensure outputs are numpy arrays of floats
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }