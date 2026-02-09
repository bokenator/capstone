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

    Flexible multi-asset order function (NO NUMBA). This function is called
    once per asset (column) per bar by the provided flexible wrapper. It
    computes the desired target position for each leg based on the z-score
    and hedge ratio, and returns the required order size to move from the
    current position to the target.

    Args:
        c: OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array of spread z-scores
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: z-score entry threshold (positive)
        exit_threshold: z-score exit threshold (usually 0)
        stop_threshold: z-score stop-loss threshold (positive)
        notional_per_leg: fixed notional in dollars per leg

    Returns:
        (size, size_type, direction) as described in prompt.
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic guards
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute basic share sizing for Asset A (positive number of shares)
    shares_a = float(notional_per_leg) / price_a
    # Ensure reasonable non-zero sizing
    if not np.isfinite(shares_a) or shares_a <= 0:
        return (np.nan, 0, 0)

    # Determine previous z to detect zero crossing
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine desired target for Asset A
    desired_a = None  # will be set to a float (target position in shares)

    # Stop-loss: close if exceeded
    if abs(z) > float(stop_threshold):
        desired_a = 0.0
    else:
        # Exit on crossing through exit_threshold (commonly 0.0)
        if not np.isnan(prev_z):
            crossed_to_exit = (prev_z > exit_threshold and z <= exit_threshold) or (
                prev_z < exit_threshold and z >= exit_threshold
            )
            if crossed_to_exit:
                desired_a = 0.0

    # If not set by stop or crossing, check entry conditions
    if desired_a is None:
        if z > float(entry_threshold):
            # Spread is high: SHORT Asset A, LONG Asset B
            desired_a = -shares_a
        elif z < -float(entry_threshold):
            # Spread is low: LONG Asset A, SHORT Asset B
            desired_a = shares_a
        else:
            # No signal -> keep existing position (no change)
            # Returning no order (np.nan) will let wrapper skip
            return (np.nan, 0, 0)

    # Compute desired target for Asset B using hedge ratio: desired_b = -hedge_ratio * desired_a
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else float(hedge_ratio[-1])
    if not np.isfinite(hr):
        # If hedge ratio not available, avoid opening/adjusting positions
        return (np.nan, 0, 0)

    desired_b = -hr * desired_a

    # Which asset are we computing order for?
    if col == 0:
        size = float(desired_a - pos_now)
    elif col == 1:
        size = float(desired_b - pos_now)
    else:
        # Unsupported column
        return (np.nan, 0, 0)

    # If size is effectively zero, do nothing
    if not np.isfinite(size) or abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return amount-sized order (number of shares)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio (rolling OLS) and z-score of the spread between two assets.

    This function is flexible with inputs: asset_a/asset_b can be either
    pandas DataFrames with a 'close' column, pandas Series, or 1D numpy arrays.

    The rolling hedge ratio at time t is computed using OLS on the window
    [t - hedge_lookback + 1, ..., t], but windows smaller than the lookback
    are allowed at the beginning (no lookahead). Rolling mean/std for the
    spread use a min_periods of 1 to avoid NaNs; std is floored to avoid
    division by zero.

    Returns a dictionary with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    All values are numpy arrays of length equal to the inputs.
    """
    # Accept either arrays/Series/DataFrames
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame input must contain 'close' column")
            return x['close'].astype(float).values
        if isinstance(x, pd.Series):
            return x.astype(float).values
        arr = np.asarray(x)
        if arr.ndim == 0:
            # scalar
            return np.array([float(arr)])
        if arr.ndim == 1:
            return arr.astype(float)
        # If 2D and has a 'close' column as second axis, attempt to use first column
        raise TypeError('Unsupported input type for price series')

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no lookahead). Use available data up to and including i.
    for i in range(n):
        start = max(0, i - int(hedge_lookback) + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Drop NaNs in the window
        mask = np.isfinite(x) & np.isfinite(y)
        xw = x[mask]
        yw = y[mask]

        if xw.size < 2:
            # Not enough data to regress: carry forward previous slope if available
            if i > 0 and np.isfinite(hedge_ratio[i - 1]):
                hedge_ratio[i] = hedge_ratio[i - 1]
            else:
                # Default to 1.0 as neutral starting hedge ratio
                hedge_ratio[i] = 1.0
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(xw, yw)
            slope = float(slope)
            if not np.isfinite(slope):
                raise ValueError('Non-finite slope')
            hedge_ratio[i] = slope
        except Exception:
            # Fallback to previous or 1.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (min_periods=1 to avoid NaNs early)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=int(zscore_lookback), min_periods=1).mean().values
    # Use ddof=0 so std defined for single-value windows (0.0)
    spread_std = spread_s.rolling(window=int(zscore_lookback), min_periods=1).std(ddof=0).values

    # Avoid division by zero
    eps = 1e-8
    spread_std = np.where(np.isfinite(spread_std), spread_std, 0.0)
    spread_std[spread_std < eps] = eps

    zscore = (spread - spread_mean) / spread_std

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }