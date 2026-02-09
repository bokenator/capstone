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

    This implementation follows the specification:
      - Entry when zscore > entry_threshold (short A, long B) or zscore < -entry_threshold (long A, short B)
      - Exit when zscore crosses exit_threshold (typically 0.0)
      - Stop-loss when |zscore| > stop_threshold
      - Position sizing: fixed notional per leg for Asset A; Asset B sized as hedge_ratio * shares_A

    Notes:
      - Flexible/no-numba mode: this function is called once per column per bar by the wrapper.
      - Returns (size, size_type, direction). Use np.nan for no action.
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(c.position_now)

    # Basic bounds checks
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if i < len(zscore) else np.nan
    h = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If key values are NaN or non-positive prices, do nothing
    if np.isnan(z) or np.isnan(h) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine share sizes based on fixed notional for Asset A and hedge ratio for Asset B
    shares_a = float(notional_per_leg / price_a)
    # Asset B shares are scaled by hedge ratio so that pos_b ~= -hedge_ratio * pos_a
    shares_b = float(h * shares_a)

    # Previous z (for cross-zero detection)
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Stop-loss: if |z| > stop_threshold -> close any existing position in this asset
    if abs(z) > stop_threshold:
        if pos != 0:
            # Close entire position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Exit on crossing the exit_threshold (typically 0.0). Detect crossing from previous bar to current.
    crossed_to_exit = False
    if not np.isnan(z_prev):
        if (z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold):
            crossed_to_exit = True

    if crossed_to_exit:
        if pos != 0:
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry logic: only enter if currently flat for this asset
    if pos == 0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B scaled by hedge ratio
                return (shares_b, 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B scaled by hedge ratio
                return (-shares_b, 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames/Series with a 'close' column or raw numpy arrays.

    Returns a dict with keys:
      - 'close_a': np.ndarray
      - 'close_b': np.ndarray
      - 'hedge_ratio': np.ndarray (rolling OLS slope computed from past data only)
      - 'zscore': np.ndarray
    """
    # Helper to extract close prices from inputs that may be arrays or DataFrames
    def _extract_close(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.DataFrame):
                if 'close' in x.columns:
                    s = x['close'].astype(float)
                else:
                    # Fallback: take first column
                    s = x.iloc[:, 0].astype(float)
            else:
                s = x.astype(float)
            return s.values
        else:
            return np.asarray(x, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Input arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no lookahead): at time i, use previous window [i-window_len, i)
    for i in range(n):
        window_len = min(i, int(hedge_lookback))
        if window_len >= 2:
            start = i - window_len
            x = close_b[start:i]
            y = close_a[start:i]
            # Remove NaNs within the window
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() >= 2:
                try:
                    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan

    # Spread using hedge_ratio aligned at time i (hedge_ratio[i] uses past data up to i-1)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score. Use min_periods=1 so zscore becomes available as soon as possible.
    spread_ser = pd.Series(spread)
    spread_mean = spread_ser.rolling(window=int(zscore_lookback), min_periods=1).mean().values
    spread_std = spread_ser.rolling(window=int(zscore_lookback), min_periods=1).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(spread[i]):
            zscore[i] = np.nan
            continue
        std = spread_std[i]
        if std is None or np.isnan(std) or std == 0:
            zscore[i] = 0.0
        else:
            zscore[i] = (spread[i] - spread_mean[i]) / std

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
