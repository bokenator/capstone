import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple, Union


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

    This is a flexible (non-numba) order function for multi-asset pairs trading.
    """
    i = int(c.i)
    col = int(c.col)

    # Safely get current position (number of shares)
    pos = getattr(c, 'position_now', 0.0)
    try:
        pos = float(pos)
    except Exception:
        pos = 0.0

    # Index bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base shares for Asset A using fixed notional per leg
    shares_a = float(notional_per_leg) / price_a
    # Asset B shares scaled by hedge ratio
    shares_b = hr * shares_a

    # Stop-loss: if |z| > stop_threshold, close positions
    if abs(z) > stop_threshold:
        if abs(pos) > 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Previous z for crossing detection
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Determine desired positions based on thresholds
    if z > entry_threshold:
        desired_a = -shares_a
        desired_b = +shares_b
    elif z < -entry_threshold:
        desired_a = +shares_a
        desired_b = -shares_b
    else:
        crossed_zero = False
        if not np.isnan(prev_z):
            if prev_z * z < 0:
                crossed_zero = True
        within_exit = abs(z) <= abs(exit_threshold)

        if crossed_zero or within_exit:
            desired_a = 0.0
            desired_b = 0.0
        else:
            return (np.nan, 0, 0)

    # Choose which asset we're ordering for
    desired = desired_a if col == 0 else desired_b

    # Compute order size as difference between desired and current position
    size = desired - pos

    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, pd.Series, np.ndarray],
    asset_b: Union[pd.DataFrame, pd.Series, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    The function is robust to inputs being DataFrames/Series with a 'close' column
    or raw numpy arrays.
    """
    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                arr = x['close'].astype(float).to_numpy(copy=False)
            else:
                arr = x.iloc[:, 0].astype(float).to_numpy(copy=False)
        elif isinstance(x, pd.Series):
            arr = x.astype(float).to_numpy(copy=False)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError("Asset A and Asset B must have the same length")

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS hedge ratio using past data only. Allow smaller windows early on
    # to avoid excessive NaNs (no lookahead). Require at least 2 points to compute regression.
    for i in range(2, n):
        start = max(0, i - hedge_lookback)
        x = close_b[start:i]
        y = close_a[start:i]

        # Remove NaNs pairwise
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data to compute slope; carry forward last value if any
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else np.nan
            continue

        x_clean = x[mask]
        y_clean = y[mask]

        # If x is constant, slope is undefined; carry forward previous slope
        if np.nanstd(x_clean) == 0:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(hedge_ratio)
    spread[valid] = close_a[valid] - hedge_ratio[valid] * close_b[valid]

    # Rolling mean/std for z-score using past windows including current spread
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=max(2, zscore_lookback // 2)).mean().to_numpy()
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=max(2, zscore_lookback // 2)).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    mask_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[mask_z] = (spread[mask_z] - spread_mean[mask_z]) / spread_std[mask_z]

    return {
        'close_a': np.array(close_a, dtype=float),
        'close_b': np.array(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
