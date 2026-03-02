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

    This function uses past indicators (no lookahead) and returns simple
    orders in the form (size, size_type, direction). It is flexible (multi-asset)
    and will be called for each asset (col=0 for Asset A, col=1 for Asset B).

    Strategy summary implemented:
    - z > entry_threshold: SHORT A, LONG B (B sized by hedge_ratio * shares_a)
    - z < -entry_threshold: LONG A, SHORT B
    - z crosses exit_threshold (e.g., 0): close both legs
    - |z| > stop_threshold: stop-loss, close both legs
    - Position sizing: shares_a = notional_per_leg / price_a
                    shares_b = shares_a * hedge_ratio

    Notes:
    - Uses only current and past values (zscore[i], hedge_ratio[i]).
    - Returns (np.nan, 0, 0) to indicate no action for this asset.
    - Returns (-c.position_now, 0, 0) to close the current position.

    Args:
        c: OrderContext-like object with attributes i (int), col (int), position_now (float)
        close_a, close_b, zscore, hedge_ratio: numpy arrays of indicators
        entry_threshold, exit_threshold, stop_threshold, notional_per_leg: params

    Returns:
        Tuple (size, size_type, direction)
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If we don't have a valid z or hedge ratio, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Sanity price checks
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Determine base shares for Asset A using fixed notional
    shares_a = notional_per_leg / price_a
    # Asset B sized by hedge ratio relative to Asset A (so pos_b ~= hedge_ratio * pos_a)
    shares_b = shares_a * hr

    # Small threshold to avoid tiny orders
    eps = 1e-8

    # Detect crossing of exit_threshold (e.g., crossing zero)
    crossed_exit = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        z_prev = float(zscore[i - 1])
        # Crossing logic: previous on one side, current on other side or equal to threshold
        if (z_prev < exit_threshold and z >= exit_threshold) or (z_prev > exit_threshold and z <= exit_threshold):
            crossed_exit = True

    # Stop-loss: absolute z exceeds stop_threshold -> close positions
    if abs(z) > stop_threshold:
        # Close this asset's position if any
        if abs(pos) > eps:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on mean reversion (crossing)
    if crossed_exit:
        if abs(pos) > eps:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry signals
    enter_short_a = z > entry_threshold
    enter_long_a = z < -entry_threshold

    # If signal to SHORT A (and LONG B)
    if enter_short_a:
        if col == 0:
            # Asset A: target = -shares_a
            target = -shares_a
            size = target - pos
            if abs(size) <= eps:
                return (np.nan, 0, 0)
            return (float(size), 0, 0)
        else:
            # Asset B: target = +shares_b
            target = float(shares_b)
            size = target - pos
            if abs(size) <= eps:
                return (np.nan, 0, 0)
            return (float(size), 0, 0)

    # If signal to LONG A (and SHORT B)
    if enter_long_a:
        if col == 0:
            target = float(shares_a)
            size = target - pos
            if abs(size) <= eps:
                return (np.nan, 0, 0)
            return (float(size), 0, 0)
        else:
            target = -float(shares_b)
            size = target - pos
            if abs(size) <= eps:
                return (np.nan, 0, 0)
            return (float(size), 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for pairs trading strategy.

    The implementation is careful to avoid lookahead bias:
    - Hedge ratio at time i is computed using an OLS regression on historical
      data up to i-1 (exclusive). If fewer than 2 historical points exist,
      hedge_ratio remains NaN for that index.
    - Z-score at time i uses spread values up to and including i (so it can be
      calculated on bar i) but the spread uses hedge_ratio computed from past
      data only, preserving causality.

    Returns a dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    All values are numpy arrays of the same length as the input.
    """
    # Accept both DataFrame/Series and numpy arrays for flexibility
    def _extract_close(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            # DataFrame with a 'close' column or a Series
            if isinstance(obj, pd.DataFrame):
                if 'close' not in obj.columns:
                    raise ValueError("asset DataFrame must contain 'close' column")
                return obj['close'].astype(float).values
            else:
                return obj.astype(float).values
        elif isinstance(obj, np.ndarray):
            return obj.astype(float)
        else:
            # Try to convert
            return np.asarray(obj, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Compute rolling hedge ratio with no lookahead: use data up to i-1
    for i in range(n):
        # build historical window up to i (exclusive)
        start = max(0, i - hedge_lookback)
        end = i  # exclusive to avoid using price at i
        if end - start >= 2:
            x = close_b[start:end]
            y = close_a[start:end]
            # Remove NaNs within window
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() >= 2:
                try:
                    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio at time i and prices at time i
    # Note: hedge_ratio[i] is computed from past data only
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean/std for z-score (uses spread up to and including i)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # z-score: (spread - mean) / std
    zscore = (spread_series - spread_mean) / spread_std

    # Convert to numpy arrays
    result = {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore.values, dtype=float),
    }

    return result
