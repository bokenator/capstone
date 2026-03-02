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

    This function is written for flexible=True multi-asset mode (NO NUMBA).

    Logic:
    - Uses z-score signal to enter/exit/stop-loss a two-legged pairs trade.
    - Entry when zscore > entry_threshold: SHORT A, LONG B (scaled by hedge_ratio).
    - Entry when zscore < -entry_threshold: LONG A, SHORT B (scaled by hedge_ratio).
    - Exit when zscore crosses exit_threshold (typically 0.0) or when |zscore| > stop_threshold.
    - Position sizing: shares_a = notional_per_leg / price_a; shares_b = hedge_ratio * shares_a.

    Returns:
        (size, size_type, direction) where size_type 0=Amount (shares).
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic bounds checks
    if i < 0:
        return (np.nan, 0, 0)

    # Validate array lengths
    n = len(zscore)
    if not (0 <= i < n):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If signal undefined, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan
    if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    # Hedge ratio at this bar must be finite to size B leg
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    if not np.isfinite(hr):
        # Wait until hedge ratio is available
        return (np.nan, 0, 0)

    # Compute base shares for Asset A and scaled shares for Asset B
    shares_a = float(notional_per_leg / price_a)
    shares_b = float(hr * shares_a)

    # Small tolerance for comparing to zero
    EPS = 1e-9
    def is_zero(x: float) -> bool:
        return abs(x) < EPS

    # Previous z-score for detecting crossing
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Stop-loss: if |z| > stop_threshold -> close any open position
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        if not is_zero(pos_now):
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Exit on crossing exit_threshold (e.g., 0.0)
    if np.isfinite(z_prev):
        crossed_down = (z_prev > exit_threshold and z <= exit_threshold)
        crossed_up = (z_prev < exit_threshold and z >= exit_threshold)
        if (crossed_down or crossed_up) and not is_zero(pos_now):
            return (-pos_now, 0, 0)

    # ENTRY LOGIC
    # Z > entry_threshold: SHORT A, LONG B (hedge_ratio units)
    if z > entry_threshold:
        if col == 0:
            # Asset A: short
            if is_zero(pos_now):
                return (-shares_a, 0, 0)
            # If already short, keep; if long, close first
            if pos_now < -EPS:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)
        else:
            # Asset B: long (scaled by hedge_ratio)
            if is_zero(pos_now):
                return (shares_b, 0, 0)
            if pos_now > EPS:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)

    # Z < -entry_threshold: LONG A, SHORT B
    if z < -entry_threshold:
        if col == 0:
            # Asset A: long
            if is_zero(pos_now):
                return (shares_a, 0, 0)
            if pos_now > EPS:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)
        else:
            # Asset B: short (scaled by hedge_ratio)
            if is_zero(pos_now):
                return (-shares_b, 0, 0)
            if pos_now < -EPS:
                return (np.nan, 0, 0)
            return (-pos_now, 0, 0)

    # Otherwise: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either pandas DataFrames/Series with a 'close' column or numpy arrays.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    All arrays have the same length as the input price series and contain no lookahead.
    """
    # Extract close arrays - support several input types
    def _extract_close(x: Any, name: str) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError(f"DataFrame for {name} must contain 'close' column")
            return x['close'].to_numpy(dtype=float)
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float)
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr

    close_a = _extract_close(asset_a, 'asset_a')
    close_b = _extract_close(asset_b, 'asset_b')

    if close_a.shape != close_b.shape:
        raise ValueError('Asset price series must have the same length')

    n = len(close_a)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS using only past data (no lookahead)
    # At index i we regress on data [i-window_len, i) (exclude current bar)
    for i in range(1, n):
        window_len = min(i, max(2, hedge_lookback)) if hedge_lookback >= 2 else min(i, 2)
        # Use up to hedge_lookback past observations but ensure at least 2 points for regression
        window_len = min(i, hedge_lookback) if i >= 2 else 0
        if window_len < 2:
            # Not enough data to compute regression
            continue
        start = i - window_len
        x = close_b[start:i]
        y = close_a[start:i]
        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        # If x is constant, slope is zero
        if np.all(x == x[0]):
            # If both constant, slope 0
            hedge_ratio[i] = 0.0
            continue
        slope, _, _, _, _ = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Spread: use hedge_ratio at time i to compute spread at time i
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (use past+current window computed without lookahead)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to avoid small-sample instability
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Where std is zero (no variation), set zscore to 0 if spread is finite
    zero_std_mask = (spread_std == 0) & np.isfinite(spread)
    zscore = np.where(zero_std_mask, 0.0, zscore)

    # Ensure output arrays are 1D numpy arrays of correct length
    close_a_out = np.asarray(close_a, dtype=float)
    close_b_out = np.asarray(close_b, dtype=float)
    hedge_ratio_out = np.asarray(hedge_ratio, dtype=float)
    zscore_out = np.asarray(zscore, dtype=float)

    return {
        'close_a': close_a_out,
        'close_b': close_b_out,
        'hedge_ratio': hedge_ratio_out,
        'zscore': zscore_out,
    }
