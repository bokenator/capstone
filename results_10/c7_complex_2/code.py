import numpy as np
import pandas as pd
try:
    import vectorbt as vbt  # Optional import; wrapped to avoid hard failure if unavailable
except Exception:
    vbt = None
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
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Pull current indicators/prices
    try:
        z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    # If z is nan, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Prices must be positive
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio must be finite for sizing
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio) and np.isfinite(hedge_ratio[i])) else np.nan
    if np.isnan(hr):
        # Without hedge ratio we cannot size the B leg correctly; skip actions
        return (np.nan, 0, 0)

    # Determine base share size using fixed notional per leg
    # Base shares for Asset A (can be fractional)
    shares_a = float(notional_per_leg) / price_a
    # Asset B shares scaled by hedge ratio (apply hedge ratio to share count)
    shares_b = shares_a * hr

    # Standardize to floats
    shares_a = float(shares_a)
    shares_b = float(shares_b)

    # Stop-loss: if |z| exceeds stop_threshold, close any open position for this asset
    if np.abs(z) > stop_threshold:
        if pos != 0.0:
            # Close entire position
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on mean reversion: z-score crosses zero (sign change)
    if i > 0:
        prev_z = zscore[i - 1]
        if np.isfinite(prev_z) and prev_z * z < 0:
            if pos != 0.0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Entry logic
    # Long A / Short B when z < -entry_threshold
    if z < -abs(entry_threshold):
        if col == 0:
            # Asset A: go long
            if pos == 0.0:
                return (shares_a, 0, 0)
            # If already in opposite position, let other bar's logic handle flipping
            return (np.nan, 0, 0)
        else:
            # Asset B: go short hedge_ratio * shares_a
            if pos == 0.0:
                return (-shares_b, 0, 0)
            return (np.nan, 0, 0)

    # Short A / Long B when z > entry_threshold
    if z > abs(entry_threshold):
        if col == 0:
            # Asset A: go short
            if pos == 0.0:
                return (-shares_a, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: go long hedge_ratio * shares_a
            if pos == 0.0:
                return (shares_b, 0, 0)
            return (np.nan, 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
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
    # Support both DataFrame inputs and raw numpy arrays (the backtester may pass either)
    # Extract close price series safely
    def _extract_close(obj):
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return obj['close'].astype(float).to_numpy(copy=True)
        if isinstance(obj, pd.Series):
            return obj.astype(float).to_numpy(copy=True)
        if isinstance(obj, np.ndarray):
            arr = obj.astype(float)
            if arr.ndim != 1:
                raise ValueError('Input numpy array must be 1D close prices')
            return arr
        # Try to coerce other iterable types
        try:
            return np.asarray(obj, dtype=float)
        except Exception:
            raise ValueError('Unsupported input type for price data')

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset A and Asset B must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each time i, regress y (A) on x (B) using up to hedge_lookback most recent observations
    # Use data up to and including current bar (no lookahead). Use smaller windows at the start (expanding).
    for i in range(n):
        # Determine window size (at least 2 points needed for linregress)
        w = min(hedge_lookback, i + 1)
        if w < 2:
            continue
        start = i - w + 1
        end = i + 1
        y = close_a[start:end]
        x = close_b[start:end]
        # Remove pairs with NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            # In case regression fails, leave as NaN
            hedge_ratio[i] = np.nan

    # Compute spread using hedge ratio (elementwise)
    spread = np.full(n, np.nan, dtype=float)
    # Only compute spread where hedge_ratio and prices are finite
    valid_mask = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[valid_mask] = close_a[valid_mask] - hedge_ratio[valid_mask] * close_b[valid_mask]

    # Rolling statistics for z-score
    spread_ser = pd.Series(spread)
    # Use min_periods=1 to avoid long NaN tails; clamp std to a small positive value to avoid division by zero
    spread_mean = spread_ser.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use ddof=0 (population std) so that single observations produce std=0; we'll clamp it below
    spread_std = spread_ser.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Clamp tiny or zero std values to a small epsilon to avoid divide-by-zero
    eps = 1e-8
    invalid_std = (~np.isfinite(spread_std)) | (spread_std < eps)
    spread_std[invalid_std] = eps

    zscore = (spread - spread_mean) / spread_std

    # Ensure outputs are numpy arrays of floats
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }