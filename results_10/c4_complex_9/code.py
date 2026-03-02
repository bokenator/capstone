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

    This implementation is written for flexible (multi-asset) mode where the
    order function is called for each asset (col) separately via a small
    simulated OrderContext (having attributes i, col, position_now, cash_now).

    Strategy summary (per the prompt):
      - Uses precomputed zscore and rolling hedge_ratio arrays
      - Entry: zscore > entry_threshold -> SHORT A, LONG B
               zscore < -entry_threshold -> LONG A, SHORT B
      - Exit: zscore crosses exit_threshold (typically 0.0) -> close both
      - Stop-loss: abs(zscore) > stop_threshold -> close both
      - Position sizing: fixed notional_per_leg, shares_a = notional/price_a,
                        shares_b = hedge_ratio * shares_a (keeps both legs ~equal notional)

    Returns a tuple (size, size_type, direction) as required by the runner.
    size_type: 0 = Amount (shares)
    direction: 0 = Both
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))

    # Defensive checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If prices are invalid, do nothing
    if not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)

    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # Compute base shares for Asset A to match notional per leg
    # and scale Asset B by hedge ratio so both legs have similar notional exposure
    try:
        shares_a = float(notional_per_leg / price_a) if price_a != 0 else np.nan
    except Exception:
        shares_a = np.nan

    if np.isnan(hr) or not np.isfinite(hr):
        shares_b = np.nan
    else:
        shares_b = hr * shares_a

    # Helper to determine crossing of exit_threshold (typically 0.0)
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_exit = False
    if not np.isnan(prev_z):
        # Crossing from above to at-or-below exit_threshold, or from below to at-or-above
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed_exit = True

    # Stop-loss: absolute zscore beyond stop_threshold -> close positions
    if np.isfinite(z) and abs(z) > stop_threshold:
        # Close position for this asset if any
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on crossing zero/exit_threshold
    if crossed_exit:
        if pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # ENTRY LOGIC
    # Short A, Long B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            # Asset A: want to be SHORT
            if pos == 0.0:
                # Open short position
                if not np.isnan(shares_a) and shares_a > 0:
                    return (-shares_a, 0, 0)
                return (np.nan, 0, 0)
            # If currently long, close first
            if pos > 0.0:
                return (-pos, 0, 0)
            # Already short -> no action
            return (np.nan, 0, 0)

        else:
            # Asset B: want to be LONG hedge_ratio * shares_a
            if np.isnan(shares_b) or shares_b == 0.0:
                return (np.nan, 0, 0)
            if pos == 0.0:
                return (shares_b, 0, 0)
            if pos < 0.0:
                # currently short -> close first
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Long A, Short B when z < -entry_threshold
    if z < -entry_threshold:
        if col == 0:
            # Asset A: want to be LONG
            if pos == 0.0:
                if not np.isnan(shares_a) and shares_a > 0:
                    return (shares_a, 0, 0)
                return (np.nan, 0, 0)
            if pos < 0.0:
                # currently short -> close first
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

        else:
            # Asset B: want to be SHORT
            if np.isnan(shares_b) or shares_b == 0.0:
                return (np.nan, 0, 0)
            if pos == 0.0:
                return (-shares_b, 0, 0)
            if pos > 0.0:
                # currently long -> close first
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators needed for the pairs trading strategy.

    Accepts either pandas DataFrame/Series with 'close' column or numpy arrays.

    Returns a dictionary with numpy arrays:
      - 'close_a', 'close_b': price arrays
      - 'hedge_ratio': rolling OLS slope (NaN until enough data)
      - 'zscore': spread z-score based on rolling mean/std
    """
    # Helper to extract close prices flexibly
    def _extract_close(x):
        # If DataFrame, expect 'close' column
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            arr = x['close'].astype(float).values
            return arr
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # Assume numpy-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            # Try to flatten if given as (n,1)
            arr = arr.reshape(-1)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)

    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 2:
        raise ValueError('zscore_lookback must be >= 2')

    # Rolling hedge ratio (OLS slope of A ~ B) using previous "hedge_lookback" samples
    hedge_ratio = np.full(n, np.nan)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # Mask out NaNs in the window
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread = A - hedge_ratio * B
    spread = np.full(n, np.nan)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for spread (z-score calculation)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Compute z-score, avoid division by zero
    zscore = (spread_series - spread_mean) / spread_std
    zscore_values = zscore.values

    # Where std is zero or NaN, set zscore to NaN (no reliable signal)
    std_vals = spread_std.values
    zscore_values[~np.isfinite(std_vals) | (std_vals == 0)] = np.nan

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore_values,
    }
