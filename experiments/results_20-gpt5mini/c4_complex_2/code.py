import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict


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

    Flexible (multi-asset) order function. This implementation uses a
    target-sizing approach: compute the desired number of shares for each
    asset and submit the delta between target and current position.

    Notes:
    - Uses fixed notional per leg. Asset B sizing is scaled by absolute
      hedge ratio magnitude; the sign of the trade is determined by the
      entry rule (short/long) rather than the sign of the hedge ratio.
    - Closes positions on zero-cross of z-score or on stop-loss.

    Args: See prompt.
    Returns:
        (size, size_type, direction) as described in prompt.
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation and bounds
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # If any critical value is NaN, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Avoid division by zero
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target shares (magnitudes). Use absolute hedge ratio for sizing.
    shares_a = notional_per_leg / price_a
    shares_b = notional_per_leg / price_b * abs(hr)

    # Determine previous z to detect zero crossings
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # 1) Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # 2) Exit on zero crossing: if z crosses 0 (sign change) -> close
    if not np.isnan(prev_z) and (prev_z * z) < 0:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # 3) Entry rules
    # If z > entry_threshold: Short A, Long B
    if z > entry_threshold:
        if col == 0:
            target = -shares_a
        else:
            target = +shares_b
        delta = target - pos
        # If no meaningful change, skip
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)

    # If z < -entry_threshold: Long A, Short B
    if z < -entry_threshold:
        if col == 0:
            target = +shares_a
        else:
            target = -shares_b
        delta = target - pos
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (float(delta), 0, 0)

    # 4) No action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for pairs strategy.

    Accepts either DataFrames/Series with a 'close' column or raw numpy
    arrays. Returns a dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore'.

    Args:
        asset_a: DataFrame/Series or ndarray for Asset A closes
        asset_b: DataFrame/Series or ndarray for Asset B closes
        hedge_lookback: rolling window for OLS regression
        zscore_lookback: rolling window for z-score

    Returns:
        dict with numpy arrays (length = number of bars)
    """
    # Extract close price arrays from possible input types
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("asset DataFrame must contain 'close' column")
            return x['close'].values.astype(float)
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # Assume array-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: slope of regressing A ~ B over hedge_lookback window
    for end in range(hedge_lookback, n + 1):
        i = end - 1
        y = close_a[end - hedge_lookback:end]
        x = close_b[end - hedge_lookback:end]
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) >= 2:
            try:
                slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge ratio (elementwise)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean/std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Avoid division by zero: mark as NaN where std is zero or NaN
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
