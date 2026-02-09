import numpy as np
import pandas as pd
import vectorbt as vbt
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
        c: vectorbt OrderContext-like object with attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float)
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array of spread (1D numpy array)
        hedge_ratio: Rolling hedge ratio array (1D numpy array)
        entry_threshold: Z-score level to enter
        exit_threshold: Z-score level to exit (typically 0.0)
        stop_threshold: Z-score level for stop-loss
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)

    Notes:
        - We compute desired target positions (in shares) for both assets based on
          notional_per_leg and hedge_ratio. The relationship maintained is:
              pos_b = -hedge_ratio * pos_a
          so that positions are opposite and scaled by hedge_ratio.
        - Entry: |zscore| > entry_threshold -> open pair (short expensive, long cheap)
        - Exit: zscore crosses exit_threshold -> close both
        - Stop-loss: |zscore| > stop_threshold -> close both
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are long enough
    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If any critical value is nan, do nothing
    if not np.isfinite(z) or not np.isfinite(hr) or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    # Compute base share size for Asset A
    # Avoid division by zero
    if price_a == 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg / price_a)

    # Desired positions logic
    # When entering a pair, we set pos_a = +/- shares_a, pos_b = -hedge_ratio * pos_a
    desired_pos_a = None
    desired_pos_b = None

    # Stop-loss has highest priority: close if threshold breached
    if abs(z) > stop_threshold:
        # Close this asset's position if any
        if pos != 0.0:
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Exit on crossing exit_threshold (e.g., crossing 0.0)
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan
    if not np.isnan(prev_z):
        crossed = False
        # crossing from above to below or below to above relative to exit_threshold
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed = True
        if crossed:
            if pos != 0.0:
                return (-pos, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Entry conditions
    if z > entry_threshold:
        # Short Asset A, Long Asset B
        desired_pos_a = -shares_a
        desired_pos_b = -hr * desired_pos_a  # equals hr * shares_a (positive if hr>0)
    elif z < -entry_threshold:
        # Long Asset A, Short Asset B
        desired_pos_a = shares_a
        desired_pos_b = -hr * desired_pos_a  # negative if hr>0
    else:
        # No action if no entry/exit/stop condition
        return (np.nan, 0, 0)

    # Select target for this column
    target = desired_pos_a if col == 0 else desired_pos_b

    # Compute order size as difference between target and current position
    order_size = float(target - pos)

    # If order size is effectively zero, do nothing
    if np.isclose(order_size, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return number of shares to trade (Amount), allow both long and short
    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible and accepts either DataFrames/Series with a
    'close' column or raw numpy arrays of close prices.

    Args:
        asset_a: DataFrame/Series/ndarray for Asset A (expects 'close' col if DataFrame)
        asset_b: DataFrame/Series/ndarray for Asset B (expects 'close' col if DataFrame)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays

    Implementation details:
        - For each time t, hedge_ratio[t] is estimated by OLS regression of
          y = close_a[window] on x = close_b[window] using only data up to t
          (no lookahead). The window is the last `hedge_lookback` points
          (or fewer at the beginning).
        - For windows with fewer than 2 valid points, the last valid hedge
          ratio is carried forward (or 1.0 at t=0).
        - Spread = close_a - hedge_ratio * close_b
        - Z-score computed with rolling mean/std over zscore_lookback with
          min_periods=1 and population std (ddof=0). If std==0, a tiny
          epsilon is added to avoid division by zero.
    """
    # Helper to extract close price array from input
    def _to_close_array(obj) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            if 'close' in obj.columns:
                arr = obj['close'].values
            else:
                # fallback to first column
                arr = obj.iloc[:, 0].values
        elif isinstance(obj, pd.Series):
            arr = obj.values
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            # Try to convert (e.g., list-like)
            arr = np.asarray(obj)

        arr = np.asarray(arr, dtype=float).flatten()
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Input arrays must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    last_valid_slope = 1.0

    # Compute rolling hedge ratio using only past and current data (no future)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Mask invalid points
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) >= 2:
            try:
                slope, _, _, _, _ = stats.linregress(x[valid], y[valid])
                if not np.isfinite(slope):
                    slope = last_valid_slope
            except Exception:
                slope = last_valid_slope
            hedge_ratio[i] = float(slope)
            last_valid_slope = float(slope)
        else:
            # Not enough data: carry forward last slope (or default 1.0)
            hedge_ratio[i] = last_valid_slope

    # Compute spread using element-wise hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std of spread (use min_periods=1 to avoid NaNs early on)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use population std (ddof=0) so that single-value windows yield std==0
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Avoid division by zero
    eps = 1e-8
    spread_std_safe = np.where(spread_std <= 0, eps, spread_std)

    zscore = (spread - spread_mean) / spread_std_safe

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
