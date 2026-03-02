import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple, Union


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This implementation works in flexible (multi-asset) mode. It returns order
    tuples (size, size_type, direction) where size_type=0 means amount (shares).

    Logic:
    - Uses z-score and hedge ratio at current bar index c.i (no lookahead).
    - Entry: |z| > entry_threshold -> target positions:
        Asset A: -sign(z) * shares_a
        Asset B:  sign(z) * shares_a * hedge_ratio
      where shares_a = notional_per_leg / price_a
    - Exit: z crosses exit_threshold (typically 0.0) OR |z| > stop_threshold -> close positions

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(c.col)

    # Defensive retrieval of current position
    try:
        pos_now = float(c.position_now)
    except Exception:
        pos_now = 0.0

    # Basic guards
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # If any required value is NaN, skip action
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute shares for Asset A based on fixed notional per leg
    # Allow fractional shares
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg) / price_a

    # Compute target positions (in shares) for both assets
    sign_z = np.sign(z) if z != 0 else 0.0

    # Determine if we should exit due to stop-loss (abs(z) > stop)
    if abs(z) > stop_threshold:
        # Close current position for this asset if any
        if not np.isclose(pos_now, 0.0):
            return (-pos_now, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Determine if we should exit due to crossing exit_threshold (e.g. 0.0)
    prev_z = zscore[i-1] if i > 0 else np.nan
    crossed = False
    if np.isfinite(prev_z) and np.isfinite(z):
        # Crossing when previous was on one side and current is on the other side of exit_threshold
        try:
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed = True
        except Exception:
            crossed = False

    if crossed:
        if not np.isclose(pos_now, 0.0):
            return (-pos_now, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry logic: if |z| > entry_threshold -> open trade (or flip to new target)
    if abs(z) > entry_threshold and sign_z != 0:
        # Target for Asset A and Asset B
        target_a = -sign_z * shares_a
        target_b = sign_z * shares_a * hr

        # Select which asset we are processing
        if col == 0:
            # Asset A: compute delta to reach target_a
            delta = target_a - pos_now
            if np.isclose(delta, 0.0):
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)
        elif col == 1:
            # Asset B
            delta = target_b - pos_now
            if np.isclose(delta, 0.0):
                return (np.nan, 0, 0)
            return (float(delta), 0, 0)
        else:
            return (np.nan, 0, 0)

    # Otherwise: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, pd.Series, np.ndarray],
    asset_b: Union[pd.DataFrame, pd.Series, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either arrays/series of close prices or DataFrames with a 'close' column.

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    Each value is a numpy array with the same length as the input prices.

    Notes:
    - Rolling OLS hedge ratio is computed using up to `hedge_lookback` past points
      but will fall back to smaller windows at the start (no lookahead).
    - Z-score rolling mean/std uses past spread values (window length zscore_lookback).
    """
    # Extract close price arrays from inputs
    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x['close'].astype(float).values
        if isinstance(x, pd.Series):
            return x.astype(float).values
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Input price arrays must be one-dimensional')
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling / expanding OLS regression to compute hedge ratio (slope of regression y ~ x)
    # Use past data only (no lookahead). Allow smaller-than-lookback windows at the start
    hlb = int(max(1, hedge_lookback))
    for i in range(2, n):
        window_len = min(hlb, i)
        y = close_a[i - window_len:i]
        x = close_b[i - window_len:i]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread calculation: spread = A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score. Use pandas rolling with min_periods=zscore_lookback to be stable,
    # but early on (when fewer observations are available) the hedge_ratio is computed using small windows
    # so we allow computing zscore once enough spread samples exist.
    zlb = int(max(1, zscore_lookback))
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zlb, min_periods=zlb).mean().to_numpy()
    spread_std = spread_series.rolling(window=zlb, min_periods=zlb).std(ddof=0).to_numpy()

    # Compute z-score where std > 0
    zscore = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[ok] = (spread[ok] - spread_mean[ok]) / spread_std[ok]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
