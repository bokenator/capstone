import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Tuple, Dict

try:
    import vectorbt as vbt
except Exception:
    # vectorbt may not be available at import time in some environments; it's not required by these
    # functions themselves, so we safely ignore import errors here.
    vbt = None


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float)
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
        - This function is called in flexible multi-asset mode: the wrapper will
          call it once per asset (col) per bar. We compute the desired target
          position (in shares) for that asset and return the difference vs
          current position as the order size.
        - Returns (np.nan, 0, 0) to indicate no action for this asset at this bar.
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Basic guards
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])

    # If z-score is not available, no trading decision
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Current prices
    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # Hedge ratio for this bar
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # Ensure we have necessary prices for sizing. If price missing, skip.
    if col == 0 and (not np.isfinite(price_a) or price_a <= 0):
        return (np.nan, 0, 0)
    if col == 1 and (not np.isfinite(price_b) or price_b <= 0):
        return (np.nan, 0, 0)

    # Compute base share sizes from fixed notional per leg. We follow the
    # example approach: shares_a = notional / price_a, shares_b = notional / price_b * hedge_ratio
    # Note: hedge_ratio may be negative depending on regression; sign is preserved.
    shares_a = notional_per_leg / price_a if price_a > 0 and np.isfinite(price_a) else np.nan
    shares_b = (notional_per_leg / price_b) * hr if price_b > 0 and np.isfinite(price_b) and np.isfinite(hr) else np.nan

    desired_target = None  # in shares for this asset

    # 1) Stop-loss: if |zscore| > stop_threshold -> close both legs
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        desired_target = 0.0

    # 2) Exit condition: z-score crosses the exit_threshold
    if desired_target is None and np.isfinite(exit_threshold) and i > 0:
        prev_z = float(zscore[i - 1]) if np.isfinite(zscore[i - 1]) else np.nan
        if np.isfinite(prev_z):
            # Detect crossing: previous above threshold and current below or vice versa
            crossed_down = (prev_z > exit_threshold) and (z <= exit_threshold)
            crossed_up = (prev_z < exit_threshold) and (z >= exit_threshold)
            if crossed_down or crossed_up:
                desired_target = 0.0

    # 3) Entry conditions
    if desired_target is None:
        if z > entry_threshold:
            # Spread is high: short Asset A, long Asset B
            if col == 0:
                # Short Asset A
                if not np.isfinite(shares_a):
                    return (np.nan, 0, 0)
                desired_target = -shares_a
            else:
                # Long Asset B
                if not np.isfinite(shares_b):
                    return (np.nan, 0, 0)
                desired_target = shares_b

        elif z < -entry_threshold:
            # Spread is low: long Asset A, short Asset B
            if col == 0:
                if not np.isfinite(shares_a):
                    return (np.nan, 0, 0)
                desired_target = shares_a
            else:
                if not np.isfinite(shares_b):
                    return (np.nan, 0, 0)
                desired_target = -shares_b

    # If no rule applies, do nothing
    if desired_target is None:
        return (np.nan, 0, 0)

    # Compute required trade size to reach desired target from current position
    size = float(desired_target - pos_now)

    # If the required size is effectively zero, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return size in shares (Amount), allow both long and short
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or a numpy array of close prices)
        asset_b: DataFrame with 'close' column for Asset B (or a numpy array of close prices)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close price array from DataFrame/Series/ndarray
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("asset DataFrame must contain a 'close' column")
            arr = x['close'].to_numpy(dtype=float)
        elif isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float)
        else:
            arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Close price input must be one-dimensional')
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    if hedge_lookback < 2 or hedge_lookback > n:
        raise ValueError('hedge_lookback must be >= 2 and <= length of data')
    if zscore_lookback < 1 or zscore_lookback > n:
        raise ValueError('zscore_lookback must be >= 1 and <= length of data')

    # Rolling hedge ratio using OLS (slope of regression of A on B)
    hedge_ratio = np.full(n, np.nan)

    # For each window ending at index i-1 (so that hedge_ratio[i] uses prices up to i-1)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove NaNs within the window
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data points to regress
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score computation
    spread_series = pd.Series(spread)

    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to avoid NaN for small windows; min_periods ensures full window
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
