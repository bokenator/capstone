import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any


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
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    # If indices out of range, do nothing
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    h = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    # If we don't have a valid zscore or hedge ratio or prices, skip action
    if np.isnan(z) or np.isnan(h) or np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine number of shares for Asset A based on notional per leg
    # Use shares_a as notional / price_a
    shares_a = float(notional_per_leg) / price_a

    # Apply hedge ratio in units: B shares = hedge_ratio * A shares
    shares_b = shares_a * h

    # Determine the target positions (shares) for this bar
    target_a = 0.0
    target_b = 0.0

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    # Entry conditions
    elif z > entry_threshold:
        # Short A, Long B
        target_a = -shares_a
        target_b = +shares_b
    elif z < -entry_threshold:
        # Long A, Short B
        target_a = +shares_a
        target_b = -shares_b
    else:
        # Exit when z-score crosses zero (mean reversion)
        prev_z = float(zscore[i - 1]) if i - 1 >= 0 and not np.isnan(zscore[i - 1]) else np.nan
        if not np.isnan(prev_z) and (prev_z * z) < 0:
            # z crossed zero -> close positions
            target_a = 0.0
            target_b = 0.0
        else:
            # No action required
            return (np.nan, 0, 0)

    # Determine which asset we're being asked to produce an order for
    if col == 0:
        # Asset A
        size = target_a - pos
        # If the size is effectively zero, do nothing
        if np.isclose(size, 0.0):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)
    elif col == 1:
        # Asset B
        size = target_b - pos
        if np.isclose(size, 0.0):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)
    else:
        return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrame inputs with a 'close' column or raw numpy arrays/Series
    containing close prices. Returns a dict with numpy arrays for 'close_a',
    'close_b', 'hedge_ratio', and 'zscore'.
    """
    # Normalize inputs to numpy arrays
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                arr = x['close'].values.astype(float)
            else:
                # If DataFrame without 'close', attempt to convert first column
                arr = x.iloc[:, 0].values.astype(float)
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Try to coerce
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Input arrays must have the same length")

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    hedge_lookback = max(2, hedge_lookback)
    zscore_lookback = max(2, zscore_lookback)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (past-only) with expanding window until lookback is reached
    # For each time i, use data from max(0, i - hedge_lookback + 1) .. i (inclusive)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]
        # Need at least two points to regress
        if len(x) >= 2:
            # If x is constant, slope is zero
            if np.allclose(x, x[0]):
                # If y is also constant, slope 0; otherwise slope 0 is reasonable
                # Avoid division by zero in linregress
                slope = 0.0
            else:
                try:
                    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
                    if np.isnan(slope):
                        slope = 0.0
                except Exception:
                    slope = 0.0
            hedge_ratio[i] = float(slope)
        else:
            hedge_ratio[i] = np.nan

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (past-only)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=2).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=2).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        s = spread[i]
        mu = spread_mean[i]
        sigma = spread_std[i]
        if np.isnan(s) or np.isnan(mu) or np.isnan(sigma) or sigma == 0:
            zscore[i] = np.nan
        else:
            zscore[i] = (s - mu) / sigma

    # As an improvement, ensure determinism and avoid NaNs after an initial warmup:
    # If early hedge_ratio values were computed using small windows, zscore will be
    # available earlier. We keep NaNs only where insufficient data existed.

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }