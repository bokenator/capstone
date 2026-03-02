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
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))  # Current position for this asset

    # Basic validation
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i] if (zscore is not None and i < len(zscore)) else np.nan
    hr = hedge_ratio[i] if (hedge_ratio is not None and i < len(hedge_ratio)) else np.nan
    price_a = close_a[i]
    price_b = close_b[i]

    # If any critical value is NaN, skip action
    if not np.isfinite(z) or not np.isfinite(hr) or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    # Compute share sizes for each leg based on fixed notional
    # shares_a: number of shares of Asset A corresponding to notional_per_leg
    # shares_b: scaled by hedge ratio
    # Protect against zero price
    if price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg / price_a)
    shares_b = float(notional_per_leg / price_b * hr)

    # Determine previous z for crossing detection
    prev_z = zscore[i - 1] if i > 0 and (zscore is not None) else np.nan

    # Stop-loss: if |z| > stop_threshold -> close both positions
    if np.isfinite(z) and abs(z) > float(stop_threshold):
        # Close the current column position
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Exit on crossing zero: close both positions when sign flips
    crossed_zero = False
    if np.isfinite(prev_z) and np.isfinite(z):
        if prev_z * z < 0:
            crossed_zero = True

    if crossed_zero:
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Entry logic
    # If z > entry_threshold: Short A, Long B
    # If z < -entry_threshold: Long A, Short B
    target_pos = None
    if z > float(entry_threshold):
        # Short Asset A, Long Asset B (scaled by hedge ratio)
        if col == 0:
            target_pos = -shares_a
        else:
            target_pos = +shares_b
    elif z < -float(entry_threshold):
        # Long Asset A, Short Asset B
        if col == 0:
            target_pos = +shares_a
        else:
            target_pos = -shares_b
    else:
        # No entry signal -> no action
        return (np.nan, 0, 0)

    # Compute order size required to move from current position to target
    order_size = target_pos - pos

    # If order size is negligible, do nothing
    if abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Return order as absolute number of shares (size_type=0), allow both directions
    return (float(order_size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or 1D array/Series
        asset_b: DataFrame with 'close' column for Asset B or 1D array/Series
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close arrays from inputs that can be arrays, Series, or DataFrames
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        else:
            arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError('Close price input must be 1-dimensional')
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Sanity bounds for lookbacks
    hedge_lookback = int(max(1, hedge_lookback))
    zscore_lookback = int(max(1, zscore_lookback))

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression (using only past and current data => no lookahead)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if x_valid.size >= 2:
            # Compute slope (hedge ratio)
            try:
                slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
                if np.isfinite(slope):
                    hedge_ratio[i] = float(slope)
                else:
                    # fallback to previous
                    hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 0.0
            except Exception:
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 0.0
        else:
            # Not enough data: use previous hedge ratio if available, else 0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 0.0

    # Compute spread using the hedge ratio at each time (no lookahead)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score using past and current values (min_periods=1 to avoid NaNs)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use ddof=0 to allow std for single observation -> 0.0
    try:
        spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()
    except TypeError:
        # Older pandas may not accept ddof parameter in rolling.std
        spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std().to_numpy()
        # Replace NaN (from ddof=1 when window=1) with 0
        spread_std = np.nan_to_num(spread_std, nan=0.0)

    # Prevent division by zero: where std is zero, set zscore to 0.0
    zscore = np.zeros(n, dtype=float)
    nonzero_mask = spread_std > 0
    zscore[nonzero_mask] = (spread[nonzero_mask] - spread_mean[nonzero_mask]) / spread_std[nonzero_mask]

    # Return results as numpy arrays
    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }