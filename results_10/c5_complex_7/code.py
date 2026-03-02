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
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B

    # Defensive checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # If we don't have a valid z-score or hedge ratio, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Avoid division by zero
    if price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    # Base shares for Asset A using fixed notional per leg
    shares_a = float(notional_per_leg) / price_a

    # Apply hedge ratio to scale Asset B shares. Use absolute hedge ratio for sizing
    # and enforce that B position is opposite sign to A position when entering.
    hr_abs = float(abs(hr))
    shares_b = hr_abs * shares_a

    # Determine previous z to detect crossing
    prev_z = zscore[i - 1] if i - 1 >= 0 else np.nan

    # Default: no action
    target_a = None
    target_b = None

    # Stop-loss: |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0

    # Entry: z > entry_threshold -> short A, long B
    elif z > entry_threshold:
        target_a = -shares_a
        target_b = +shares_b

    # Entry: z < -entry_threshold -> long A, short B
    elif z < -entry_threshold:
        target_a = +shares_a
        target_b = -shares_b

    else:
        # Exit on crossing zero (or exit_threshold). We consider crossing through exit_threshold.
        if not np.isnan(prev_z):
            crossed_from_above = (prev_z > exit_threshold) and (z <= exit_threshold)
            crossed_from_below = (prev_z < exit_threshold) and (z >= exit_threshold)
            if crossed_from_above or crossed_from_below:
                target_a = 0.0
                target_b = 0.0

    # If no trading signal, return No Order
    if target_a is None or target_b is None:
        return (np.nan, 0, 0)

    # Determine current position for this asset
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Select the appropriate target for this column
    target = target_a if col == 0 else target_b

    # Compute required order size (shares)
    size = float(target - pos_now)

    # If the required size is negligible, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return number of shares to trade (size_type=0 -> Amount)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Notes:
    - Rolling OLS uses previous up to `hedge_lookback` samples. For early bars
      where fewer samples are available, it will use the available history
      (min lookback = 2) to avoid long warmup and prevent lookahead bias.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or a 1D array/Series
        asset_b: DataFrame with 'close' column for Asset B or a 1D array/Series
        hedge_lookback: Lookback for rolling OLS hedge ratio (max window size)
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close prices from inputs (DataFrame/Series/ndarray)
    def _extract_close(x: Any) -> np.ndarray:
        if hasattr(x, 'columns') and 'close' in getattr(x, 'columns'):
            return x['close'].values.astype(float)
        # If it's a DataFrame without 'close', try first column
        if hasattr(x, 'iloc') and getattr(x, 'ndim', 1) == 2:
            try:
                return x.iloc[:, 0].values.astype(float)
            except Exception:
                pass
        arr = np.asarray(x, dtype=float)
        if arr.ndim > 1:
            arr = arr.ravel()
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = close_a.shape[0]

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression using past samples (no lookahead). We allow smaller
    # windows at the beginning (min window size = 2) so that indicators become
    # available earlier than strict hedge_lookback to reduce warmup NaNs.
    for i in range(2, n):
        window = min(hedge_lookback, i)
        y = close_a[i - window:i]
        x = close_b[i - window:i]

        # Skip window if any NaNs present or insufficient variance
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue
        if np.all(x == x[0]):
            hedge_ratio[i] = np.nan
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread using current prices and hedge_ratio computed from past window
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std for z-score (includes current spread point)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score
    zscore = np.full(n, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
        zscore[idx] = (spread[idx] - spread_mean[idx]) / spread_std[idx]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
