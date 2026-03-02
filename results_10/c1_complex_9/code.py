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
        - Uses amount-based sizing (size_type=0) where size is number of shares.
        - The B-leg share sizing follows the pre-computed hedge_ratio:
            shares_b = (notional_per_leg / price_b) * hedge_ratio
          This mirrors the example structure in the prompt.
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))  # Current position for this asset

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Protect against invalid prices
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute target shares based on notional per leg
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Small tolerance for comparing positions
    zero_tol = 1e-8

    # Determine previous z for crossing detection
    z_prev = np.nan
    if i > 0:
        z_prev = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else np.nan

    # Helper to close position (return order to bring pos to zero)
    def close_order(current_pos: float):
        if abs(current_pos) < zero_tol:
            return (np.nan, 0, 0)
        return (-float(current_pos), 0, 0)

    # Exit conditions (stop-loss or z-score crossing zero)
    # Stop-loss: |z| > stop_threshold -> close both legs
    if abs(z) > float(stop_threshold):
        # If this asset has a position, close it
        return close_order(pos)

    # Crossing zero: z crosses 0 compared to previous bar -> close both legs
    if not np.isnan(z_prev):
        crossed_zero = (z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0)
        if crossed_zero:
            return close_order(pos)

    # Entry logic
    # If z > entry_threshold: Short A, Long B
    # If z < -entry_threshold: Long A, Short B
    if z > float(entry_threshold):
        # Desired: A short, B long (quantities determined by hedge ratio)
        if col == 0:
            # Asset A: Short
            desired_size = -shares_a
            # If already have opposite sign, close first
            if abs(pos) > zero_tol and np.sign(pos) != np.sign(desired_size):
                return close_order(pos)
            # If flat, enter
            if abs(pos) < zero_tol:
                return (desired_size, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: Long by shares_b (shares_b may be negative depending on hr)
            desired_size = +shares_b
            if abs(pos) > zero_tol and np.sign(pos) != np.sign(desired_size):
                return close_order(pos)
            if abs(pos) < zero_tol:
                return (desired_size, 0, 0)
            return (np.nan, 0, 0)

    if z < -float(entry_threshold):
        # Desired: A long, B short
        if col == 0:
            desired_size = +shares_a
            if abs(pos) > zero_tol and np.sign(pos) != np.sign(desired_size):
                return close_order(pos)
            if abs(pos) < zero_tol:
                return (desired_size, 0, 0)
            return (np.nan, 0, 0)
        else:
            desired_size = -shares_b
            if abs(pos) > zero_tol and np.sign(pos) != np.sign(desired_size):
                return close_order(pos)
            if abs(pos) < zero_tol:
                return (desired_size, 0, 0)
            return (np.nan, 0, 0)

    # No action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function accepts either pandas DataFrames with a 'close' column or raw
    numpy arrays / pandas Series representing close prices.

    Args:
        asset_a: DataFrame or 1D array-like with 'close' prices for Asset A
        asset_b: DataFrame or 1D array-like with 'close' prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with keys 'close_a', 'close_b', 'hedge_ratio', 'zscore' mapping to
        1D numpy arrays of the same length as the input price series.
    """
    # Extract close price arrays from inputs (support multiple input types)
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame input must have a 'close' column")
            return x['close'].astype(float).to_numpy()
        if isinstance(x, pd.Series):
            return x.astype(float).to_numpy()
        x_arr = np.asarray(x, dtype=float)
        # If a 2D array (e.g., single-column DataFrame values), try to flatten
        if x_arr.ndim == 2 and x_arr.shape[1] == 1:
            x_arr = x_arr.ravel()
        if x_arr.ndim != 1:
            raise ValueError("Input must be 1D array-like or a DataFrame/Series with a single 'close' column")
        return x_arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression to estimate hedge ratio (slope)
    # Assign hedge_ratio at index i using data from [i-hedge_lookback : i]
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove NaNs in the window
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue

        x_clean = x[mask]
        y_clean = y[mask]

        # If x is constant, linregress may fail or produce NaN slope; handle gracefully
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            if np.isfinite(slope):
                hedge_ratio[i] = float(slope)
            else:
                hedge_ratio[i] = np.nan
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using the rolling hedge ratio
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score (use pandas rolling to handle NaNs)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score
    zscore = np.full(n, np.nan, dtype=float)
    valid_stats = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_stats] = (spread[valid_stats] - spread_mean[valid_stats]) / spread_std[valid_stats]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }