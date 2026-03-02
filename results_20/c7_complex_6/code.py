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

    This implementation is flexible (multi-asset) and non-numba.
    See prompt for detailed behavior.
    """
    # Current bar and column
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validations
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # If zscore is not available, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Hedge ratio must be available
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if not np.isfinite(hr):
        # If hedge ratio is not available, avoid entering
        # Allow exits based on z-score/stop even if hr missing
        hr = np.nan

    # Compute previous z-score for exit-on-cross detection
    z_prev = zscore[i - 1] if i > 0 else np.nan

    # Stop-loss: if magnitude exceeds stop_threshold, close any position
    if np.isfinite(z) and abs(z) > stop_threshold:
        if abs(pos) > 1e-12:
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Exit when z-score crosses the exit_threshold
    if i > 0 and np.isfinite(z_prev) and np.isfinite(z):
        crossed_down = (z_prev > exit_threshold and z <= exit_threshold)
        crossed_up = (z_prev < exit_threshold and z >= exit_threshold)
        if crossed_down or crossed_up:
            if abs(pos) > 1e-12:
                return (-pos, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Determine target shares based on notional sizing and hedge ratio
    # Primary rule: use notional_per_leg to size Asset A, then scale Asset B by hedge ratio
    # shares_a: number of Asset A units
    if price_a <= 0 or price_b <= 0 or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    shares_a = notional_per_leg / price_a
    # If hedge ratio is not finite, avoid entering
    if np.isfinite(hr):
        shares_b = float(hr) * shares_a
    else:
        shares_b = np.nan

    # Entry logic
    # Long A / Short B when z < -entry_threshold
    if z < -entry_threshold:
        if col == 0:
            # Asset A: enter long if flat
            if abs(pos) < 1e-12:
                return (shares_a, 0, 0)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: enter short (opposite sign) if flat
            if not np.isfinite(shares_b):
                return (np.nan, 0, 0)
            if abs(pos) < 1e-12:
                return (-shares_b, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Short A / Long B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            # Asset A: enter short if flat
            if abs(pos) < 1e-12:
                return (-shares_a, 0, 0)
            else:
                return (np.nan, 0, 0)
        else:
            # Asset B: enter long if flat
            if not np.isfinite(shares_b):
                return (np.nan, 0, 0)
            if abs(pos) < 1e-12:
                return (shares_b, 0, 0)
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

    Accepts pd.DataFrame inputs with a 'close' column, or raw numpy arrays.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    All arrays have the same length as the input price series.
    """
    # Helper to extract close prices from DataFrame/Series/ndarray
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return x['close'].astype(float).values
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # Assume array-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError('close price input must be 1D')
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS: for each time i, regress y=close_a on x=close_b using up to hedge_lookback most recent points
    # Use expanding windows at start (min 2 valid points) to avoid long warmup NaNs
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x_win = close_b[start:i + 1]
        y_win = close_a[start:i + 1]
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if np.count_nonzero(mask) >= 2:
            x_valid = x_win[mask]
            y_valid = y_win[mask]
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread: Price_A - hedge_ratio * Price_B
    spread = np.full(n, np.nan)
    for i in range(n):
        if np.isfinite(hedge_ratio[i]) and np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
            spread[i] = close_a[i] - hedge_ratio[i] * close_b[i]
        else:
            spread[i] = np.nan

    # Rolling mean and std for z-score (use min_periods=1 to reduce warmup NaNs)
    spread_s = pd.Series(spread)
    roll_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use ddof=0 so that std of a single value is 0 rather than NaN
    roll_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    for i in range(n):
        s = spread[i]
        m = roll_mean[i]
        sd = roll_std[i]
        if not np.isfinite(s):
            # If current spread is NaN but mean/std exist, avoid lookahead - keep NaN
            zscore[i] = np.nan
        else:
            if not np.isfinite(sd) or sd == 0:
                # If no variation, set z-score to 0
                zscore[i] = 0.0
            else:
                zscore[i] = (s - m) / sd

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
