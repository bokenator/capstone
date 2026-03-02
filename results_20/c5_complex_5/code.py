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
    """
    i = int(c.i)
    col = int(c.col)

    # Basic sanity checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hedge = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If insufficient data, do nothing
    if np.isnan(z) or np.isnan(hedge):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Determine base share size for Asset A (fixed notional per leg)
    shares_a = notional_per_leg / price_a
    # Asset B is scaled by hedge ratio (number of shares)
    shares_b = hedge * shares_a

    # Determine previous z to detect crossing of exit threshold
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine target positions (in shares)
    target_a = None
    target_b = None

    # Stop-loss takes highest priority
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0

    # Exit when z-score crosses the exit_threshold (e.g., 0.0)
    elif i > 0 and not np.isnan(z_prev):
        crossed_down = (z_prev > exit_threshold) and (z <= exit_threshold)
        crossed_up = (z_prev < exit_threshold) and (z >= exit_threshold)
        if crossed_down or crossed_up:
            target_a = 0.0
            target_b = 0.0

    # Entry conditions
    if target_a is None:
        if z > entry_threshold:
            # Short A, Long B
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +shares_a
            target_b = -shares_b

    # If still None => no action (maintain current positions)
    if target_a is None or target_b is None:
        return (np.nan, 0, 0)

    # Compute order size for this column: target - current position
    pos_now = float(getattr(c, 'position_now', 0.0))

    if col == 0:
        size = target_a - pos_now
    else:
        size = target_b - pos_now

    # If change is negligible, do nothing
    if np.isclose(size, 0.0):
        return (np.nan, 0, 0)

    # Return amount-based order (number of shares)
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
        asset_a: DataFrame with 'close' column for Asset A OR a numpy array/series
        asset_b: DataFrame with 'close' column for Asset B OR a numpy array/series
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close price arrays from inputs that may be DataFrames/Series/ndarrays
    def _to_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x['close'].values.astype(float)
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        if isinstance(x, np.ndarray):
            return x.astype(float)
        # Try to convert generic sequence
        return np.asarray(x, dtype=float)

    close_a = _to_array(asset_a)
    close_b = _to_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Input arrays must have the same length')

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (using inclusive window up to current index) with at least 2 samples
    for i in range(n):
        # Inclusive window [start, i]
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Remove pairs where either is NaN
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            # Not enough data points to estimate
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        # Catch degenerate cases
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            if np.isfinite(slope):
                hedge_ratio[i] = float(slope)
        except Exception:
            # In case regression fails, leave as NaN
            continue

    # Compute spread using hedge_ratio (elementwise)
    spread = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(hedge_ratio[i]) or np.isnan(close_a[i]) or np.isnan(close_b[i]):
            spread[i] = np.nan
        else:
            spread[i] = close_a[i] - hedge_ratio[i] * close_b[i]

    # Rolling mean and std for z-score. Use min_periods=1 to avoid long warmups,
    # and population std (ddof=0) so std is 0 when only one valid observation.
    spread_s = pd.Series(spread)

    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score, handling zero std gracefully
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(spread[i]) or np.isnan(spread_mean[i]) or np.isnan(spread_std[i]):
            zscore[i] = np.nan
        elif spread_std[i] == 0 or not np.isfinite(spread_std[i]):
            # No variation -> z-score 0
            zscore[i] = 0.0
        else:
            zscore[i] = float((spread[i] - spread_mean[i]) / spread_std[i])

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }