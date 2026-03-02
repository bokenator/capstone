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

    Logic:
    - Determine target positions (in shares) based on z-score and hedge ratio.
    - Size Asset A using notional_per_leg / price_a as base share count.
    - Asset B is sized to maintain the hedge ratio relative to Asset A
      such that position_b ~= -hedge_ratio * position_a (i.e., scaled and
      generally opposite sign when hedge_ratio > 0).
    - Entry when |z| > entry_threshold (sign determines direction).
    - Exit when z crosses exit_threshold (default 0.0) or |z| > stop_threshold.

    Returns (size, size_type, direction) where size is in shares (size_type=0).
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))  # current position in shares for this asset

    # Safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Read current z and hedge ratio
    z = float(zscore[i]) if i < len(zscore) else np.nan
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio) and not np.isnan(hedge_ratio[i])) else 1.0

    # If z is nan, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Base share sizing for Asset A (positive number)
    shares_a_base = float(notional_per_leg) / price_a

    # Determine previous z to detect crossing of exit threshold
    prev_z = float(zscore[i - 1]) if i > 0 and i - 1 < len(zscore) and not np.isnan(zscore[i - 1]) else np.nan

    # Determine if we should exit due to stop-loss
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Detect crossing of exit_threshold (default 0.0)
        crossed = False
        if not np.isnan(prev_z):
            # Crossing when moving from one side of the threshold to the other (inclusive)
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed = True

        if crossed:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry signals
            if z > entry_threshold:
                # Short Asset A, Long Asset B (scaled by hedge ratio)
                target_a = -shares_a_base
                # We set Asset B to be scaled so that position_b ~= -hedge_ratio * position_a
                target_b = -hr * target_a
            elif z < -entry_threshold:
                # Long Asset A, Short Asset B (scaled by hedge ratio)
                target_a = shares_a_base
                target_b = -hr * target_a
            else:
                # No change to positions
                return (np.nan, 0, 0)

    # Determine order for the current column: return delta in shares (size_type=0)
    if col == 0:
        # Asset A
        delta = target_a - pos
    else:
        # Asset B
        delta = target_b - pos

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return number of shares to trade (positive=buy, negative=sell)
    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrame/Series with a 'close' column or raw numpy arrays.

    Returns a dict with the following numpy arrays (float64):
      - 'close_a'
      - 'close_b'
      - 'hedge_ratio'   (rolling OLS slope)
      - 'zscore'        (rolling z-score of spread)

    Implementation notes:
      - Rolling OLS uses available data up to and including the current bar
        with a maximum window of `hedge_lookback` (causal - no lookahead).
      - For early bars where the full hedge_lookback is not available, the
        regression is run on the available history (>=2 points). The first
        bar (index 0) where regression is impossible falls back to 1.0.
      - Rolling mean/std for z-score use min_periods=1 to avoid NaNs for
        short series; z-score is set to 0 where std == 0.
    """
    # Helper to extract close arrays from possible input types
    def _extract_close(x):
        # Numpy array
        if isinstance(x, np.ndarray):
            return x.astype(float)
        # Pandas DataFrame or Series
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                return x['close'].astype(float).values
            # If no 'close' column, try first column
            return x.iloc[:, 0].astype(float).values
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # Fallback - try to convert
        return np.asarray(x, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Input series must have the same length")

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)
    last_valid = 1.0

    # Rolling OLS (causal): for each time i, regress y=A and x=B on window [i - lookback + 1, i]
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        y = close_a[start:i + 1]
        x = close_b[start:i + 1]

        # Need at least two non-NaN points to run regression
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            # If x has zero variance, linregress may return nan or inf; guard with try/except
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                if not np.isfinite(slope):
                    slope = last_valid
                hedge_ratio[i] = float(slope)
                last_valid = hedge_ratio[i]
            except Exception:
                hedge_ratio[i] = last_valid
        else:
            # Not enough data - carry forward last valid or default 1.0
            hedge_ratio[i] = last_valid

    # Compute spread using hedge ratio (elementwise)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (causal). Use min_periods=1 to avoid long NaN tails.
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use population std (ddof=0) to keep values defined early
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Avoid division by zero: where std == 0 set zscore to 0.0
    zscore = np.full(n, np.nan, dtype=float)
    denom = spread_std.copy()
    zero_mask = (denom == 0) | (~np.isfinite(denom))

    # Compute where denom > 0
    valid_mask = (~zero_mask) & np.isfinite(spread) & np.isfinite(spread_mean)
    zscore[valid_mask] = (spread[valid_mask] - spread_mean[valid_mask]) / denom[valid_mask]
    # For zero std (no variation), set zscore to 0 (no signal)
    zscore[zero_mask & np.isfinite(spread)] = 0.0

    return {
        'close_a': close_a.astype(float),
        'close_b': close_b.astype(float),
        'hedge_ratio': hedge_ratio.astype(float),
        'zscore': zscore.astype(float),
    }