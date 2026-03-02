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

    Flexible multi-asset order function (NO NUMBA).

    Args and return as specified in the prompt.
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0) or 0.0)

    # Basic bounds check
    if i < 0 or i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If critical values are NaN, skip
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Previous z for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Compute target positions (number of shares) for both assets
    target_a = None
    target_b = None

    # Stop-loss: if |z| > stop_threshold -> close positions
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0

    # Exit on crossing exit_threshold (e.g., crossing zero)
    elif (not np.isnan(prev_z)) and (
        (prev_z > exit_threshold and z <= exit_threshold) or
        (prev_z < exit_threshold and z >= exit_threshold)
    ):
        target_a = 0.0
        target_b = 0.0

    # Entry signals
    elif z > entry_threshold:
        # Short A, Long B (scaled by hedge ratio)
        shares_a = notional_per_leg / price_a
        shares_b = hr * shares_a
        target_a = -shares_a
        target_b = shares_b

    elif z < -entry_threshold:
        # Long A, Short B (scaled by hedge ratio)
        shares_a = notional_per_leg / price_a
        shares_b = hr * shares_a
        target_a = shares_a
        target_b = -shares_b

    else:
        # No signal -> no action
        return (np.nan, 0, 0)

    # Determine this column's target and compute delta from current position
    target = target_a if col == 0 else target_b
    size = float(target - pos_now)

    # If size is effectively zero, do nothing
    if abs(size) < 1e-12:
        return (np.nan, 0, 0)

    # Return size in shares (size_type=0), allow both long and short (direction=0)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is robust to being passed either DataFrames/Series with a
    'close' column or raw numpy arrays of close prices (the runner passes
    numpy arrays). It returns dictionaries of numpy arrays with keys:
    'close_a', 'close_b', 'hedge_ratio', 'zscore'.

    Hedge ratio is computed via rolling OLS (slope) with lookback up to
    `hedge_lookback` using only past and current data. When there are fewer
    than 2 points, the hedge ratio is NaN. For earlier indices with fewer
    than hedge_lookback points, the regression uses the available points
    (min_periods=2).

    Z-score uses a rolling mean/std of the spread with window `zscore_lookback`.
    We use min_periods=2 for early stability and guard against zero std.
    """
    # Accept either DataFrame/Series with 'close' column or numpy arrays
    def _extract_close(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.DataFrame) and 'close' in x.columns:
                return x['close'].values
            # If Series or no 'close' column, try to use values directly
            return x.values if hasattr(x, 'values') else np.asarray(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            # Try to coerce
            return np.asarray(x)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    # Ensure 1-D arrays
    close_a = np.asarray(close_a).flatten()
    close_b = np.asarray(close_b).flatten()

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each time i compute slope of regression of y(close_a)
    # on x(close_b) using up to `hedge_lookback` most recent observations
    # (including current point). Require at least 2 points.
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        end = i + 1  # inclusive of current index
        if end - start < 2:
            # Not enough points to estimate slope
            hedge_ratio[i] = np.nan
            continue

        x = close_b[start:end]
        y = close_a[start:end]

        # If any NaNs in window, skip estimation
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If constant x, slope is undefined -> set slope to 0 to avoid NaNs
        if np.allclose(x, x[0]):
            hedge_ratio[i] = 0.0
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    # Compute spread using the hedge ratio (elementwise). Where hedge_ratio is NaN,
    # spread will be NaN as well.
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean/std for z-score. Use pandas rolling with min_periods=2 to avoid
    # excessive NaNs early. Use ddof=0 for population std to be numerically stable.
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=2).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=2).std(ddof=0).to_numpy()

    # Compute zscore safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std))
    # Guard against zero std
    eps = 1e-8
    zscore[valid] = (spread[valid] - spread_mean[valid]) / (spread_std[valid] + eps)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
