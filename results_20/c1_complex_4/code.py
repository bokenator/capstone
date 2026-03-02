import numpy as np
import pandas as pd
from scipy import stats

# Try to import vectorbt if available; keep module import safe if not present
try:
    import vectorbt as vbt
except Exception:
    vbt = None


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

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(np.nan_to_num(c.position_now, nan=0.0))  # Current position for this asset

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If prices invalid, skip
    if not np.isfinite(price_a) or price_a <= 0 or not np.isfinite(price_b) or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio at this bar
    hr = hedge_ratio[i] if (i < len(hedge_ratio)) else np.nan
    hr = float(hr) if np.isfinite(hr) else np.nan

    # Determine raw share amounts based on fixed notional per leg
    # shares_a: how many shares of A to trade for notional_per_leg
    # shares_b: scaled by hedge ratio (may be negative if hedge ratio is negative)
    shares_a = notional_per_leg / price_a if price_a > 0 else np.nan
    shares_b = (notional_per_leg / price_b * hr) if (price_b > 0 and not np.isnan(hr)) else np.nan

    # Helper: check if crossing exit threshold between previous and current bar
    def _crossed(prev: float, curr: float, threshold: float) -> bool:
        if np.isnan(prev) or np.isnan(curr):
            return False
        # Crossed threshold from above to below or below to above
        return (prev > threshold and curr <= threshold) or (prev < threshold and curr >= threshold)

    # STOP-LOSS: if |z| > stop_threshold -> close if we have a position
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        if pos != 0.0:
            # Close entire position for this asset
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # EXIT: z-score crossed the exit_threshold (e.g., 0.0)
    if i > 0:
        prev_z = zscore[i - 1]
        if _crossed(prev_z, z, exit_threshold):
            if pos != 0.0:
                return (-pos, 0, 0)
            else:
                return (np.nan, 0, 0)

    # ENTRY: open positions when z crosses entry thresholds
    # Long/short sizing uses number of shares (size_type = 0)
    # For z > entry_threshold: Short A, Long B (hedge_ratio units)
    if z > entry_threshold:
        # Need a valid hedge ratio to size B leg
        if np.isnan(shares_b):
            return (np.nan, 0, 0)

        if col == 0:
            # Asset A: short shares_a
            if pos == 0.0:
                return (-shares_a, 0, 0)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: long shares_b (shares_b may be negative if hr < 0)
            if pos == 0.0:
                return (shares_b, 0, 0)
            else:
                return (np.nan, 0, 0)

    # For z < -entry_threshold: Long A, Short B (hedge_ratio units)
    if z < -entry_threshold:
        if np.isnan(shares_b):
            return (np.nan, 0, 0)

        if col == 0:
            # Asset A: long shares_a
            if pos == 0.0:
                return (shares_a, 0, 0)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: short shares_b (invert sign)
            if pos == 0.0:
                return (-shares_b, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a 1D numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR a 1D numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close price array from DataFrame or ndarray/Series
    def _to_close_array(x) -> np.ndarray:
        # If it's a DataFrame or Series, try to read 'close' column
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.DataFrame):
                if 'close' in x.columns:
                    arr = x['close'].values
                else:
                    # If DataFrame without 'close', try to flatten
                    arr = x.values.flatten()
            else:
                # Series
                arr = x.values
        else:
            # Assume numpy-like
            arr = np.asarray(x)
            if arr.ndim > 1:
                # Flatten to 1D if possible
                arr = arr.ravel()
        return arr.astype(float)

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError('Asset A and Asset B must have the same length')

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression to compute hedge ratio (slope of regression of A on B)
    # For each window ending at index i (exclusive), set hedge_ratio[i] = slope of regression on previous window
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be at least 2')

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x is constant, slope is undefined
        if np.isclose(np.std(x), 0.0):
            hedge_ratio[i] = np.nan
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = slope

    # Compute spread: spread = A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    # Only compute where hedge_ratio is finite
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Z-score: (spread - mean) / std
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std != 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
