import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Union, Dict


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
        c: vectorbt OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter
        exit_threshold: Z-score level to exit
        stop_threshold: Z-score level for stop-loss
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction) tuple as required by vectorbt flexible order func.
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # If indicator unavailable, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(close_a[i]) or np.isnan(close_b[i]):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Defensive: avoid division by zero
    if price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    # Base number of shares for Asset A sized by notional per leg
    shares_a = notional_per_leg / price_a

    # Determine previous z for exit (crossing zero) detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Stop-loss: absolute z-score exceeds stop_threshold -> close positions
    if abs(z) > stop_threshold:
        # Close if we have a position; otherwise no action
        if abs(pos) > 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on crossing exit_threshold (typically 0.0)
    crossed = False
    if not np.isnan(prev_z):
        # Detect crossing: previous above and now below or previous below and now above
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed = True
    # Also if z is exactly at the exit threshold
    if not crossed and np.isclose(z, exit_threshold):
        crossed = True

    if crossed:
        if abs(pos) > 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic: only enter when |z| exceeds entry_threshold
    if z > entry_threshold:
        # Short Asset A, Long Asset B
        target_a = -shares_a
        target_b = -hr * target_a  # Ensures position_b ~= -hedge_ratio * position_a
    elif z < -entry_threshold:
        # Long Asset A, Short Asset B
        target_a = shares_a
        target_b = -hr * target_a
    else:
        # No entry signal
        return (np.nan, 0, 0)

    # Determine which asset we're handling and compute delta from current position
    if col == 0:
        # Asset A
        delta = float(target_a - pos)
    elif col == 1:
        # Asset B
        delta = float(target_b - pos)
    else:
        return (np.nan, 0, 0)

    # If delta is effectively zero, do nothing
    if np.isclose(delta, 0.0):
        return (np.nan, 0, 0)

    # Return order in shares (Amount)
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with 'close' column or 1D numpy arrays of close prices.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    Each value is a 1D numpy array of the same length as input prices.
    """
    # Extract close series depending on input types
    def extract_close(x, name: str) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError(f"DataFrame for {name} must contain 'close' column.")
            return x['close'].astype(float).values
        if isinstance(x, pd.Series):
            return x.astype(float).values
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            # If passed 2D single-column array, try to flatten
            try:
                arr = arr.ravel()
            except Exception:
                raise ValueError(f"Unsupported input shape for {name}: {np.shape(x)}")
        return arr

    close_a = extract_close(asset_a, 'asset_a')
    close_b = extract_close(asset_b, 'asset_b')

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan)

    # Rolling (expanding) OLS regression to compute hedge ratio (slope of regressing A on B)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for end in range(n):
        start = max(0, end - hedge_lookback + 1)
        y = close_a[start:end + 1]
        x = close_b[start:end + 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[end] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[end] = float(slope)
        except Exception:
            hedge_ratio[end] = np.nan

    # Compute spread and z-score using rolling mean/std (no lookahead)
    spread = np.full(n, np.nan)
    for i in range(n):
        if np.isfinite(hedge_ratio[i]) and np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
            spread[i] = close_a[i] - hedge_ratio[i] * close_b[i]
        else:
            spread[i] = np.nan

    spread_series = pd.Series(spread)
    # Use min_periods=1 to ensure values are produced as soon as there's data (avoids NaNs after warmup)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
