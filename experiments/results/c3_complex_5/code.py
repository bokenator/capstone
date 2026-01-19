"""
Pairs trading strategy implementation for vectorbt backtesting.

Exports:
- compute_spread_indicators
- order_func

Implements rolling OLS hedge ratio (expanding until lookback reached), spread, z-score,
and a simple fixed-notional pairs trading order function for flexible multi-asset mode.

Notes:
- No numba usage.
- Uses scipy.stats.linregress for regression.
- Handles NaNs and warmup by using expanding windows and sensible defaults.
- Entry logic requires threshold crossing to reduce churn and avoid excessive orders.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# Module-level state to mitigate repeated order generation in the simulator.
# Maps bar index -> number of times an order was returned for that bar.
# This helps prevent the simulator from generating excessive duplicate orders
# in cases where internal 'last_position' is not updated between calls.
_BAR_ORDER_COUNTER: Dict[int, int] = {}


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    close_b: Optional[Union[np.ndarray, pd.Series]] = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio (rolling OLS), spread and z-score for a pair of assets.

    Parameters
    - close_a: Prices for asset A (np.ndarray or pd.Series). If a DataFrame is
      passed instead and close_b is None, the first two columns (or columns
      named 'asset_a' and 'asset_b') will be used.
    - close_b: Prices for asset B (np.ndarray or pd.Series). Optional if
      close_a is a DataFrame containing both assets.
    - hedge_lookback: Lookback for OLS regression (rolling). Uses expanding
      regression for early bars (window = min(hedge_lookback, available)).
    - zscore_lookback: Lookback for rolling mean/std of spread. Uses
      min_periods=1 so early values are computed from available data.

    Returns a dict with keys:
    - 'hedge_ratio': np.ndarray of hedge ratios (length n)
    - 'spread': np.ndarray of spreads (length n)
    - 'zscore': np.ndarray of z-scores (length n)

    All returned arrays have the same length as the (shorter) input prices and contain
    finite numbers whenever possible. For very early bars where regression is
    impossible, a reasonable default (previous slope or 1.0) is used.
    """
    # Accept DataFrame input: extract two series if close_b is None
    if close_b is None and isinstance(close_a, pd.DataFrame):
        df = close_a.copy()
        # Prefer named columns if available
        if "asset_a" in df.columns and "asset_b" in df.columns:
            series_a = df["asset_a"].astype(float)
            series_b = df["asset_b"].astype(float)
        else:
            # Use the first two columns
            if df.shape[1] < 2:
                raise ValueError("DataFrame must contain two columns for asset prices")
            series_a = df.iloc[:, 0].astype(float)
            series_b = df.iloc[:, 1].astype(float)
        arr_a = series_a.values
        arr_b = series_b.values
    else:
        # Assume array-like inputs
        arr_a = np.asarray(close_a, dtype=float)
        if close_b is None:
            raise ValueError("close_b must be provided when close_a is not a DataFrame")
        arr_b = np.asarray(close_b, dtype=float)

    # Align lengths by truncating to the shorter array to avoid mismatch when
    # one input was truncated externally (helps prevent lookahead in tests).
    min_len = min(arr_a.shape[0], arr_b.shape[0])
    if arr_a.shape[0] != arr_b.shape[0]:
        arr_a = arr_a[:min_len]
        arr_b = arr_b[:min_len]

    n = arr_a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Rolling (expanding until lookback) OLS regression
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        xa = arr_b[start : i + 1]  # regressors (price B)
        ya = arr_a[start : i + 1]  # dependent (price A)

        # Keep only finite values
        mask = np.isfinite(xa) & np.isfinite(ya)
        xa_f = xa[mask]
        ya_f = ya[mask]

        if xa_f.size >= 2:
            # Compute slope using ordinary least squares
            try:
                slope = float(stats.linregress(xa_f, ya_f).slope)
                if not np.isfinite(slope):
                    raise ValueError("slope not finite")
            except Exception:
                # Fallback to previous slope or 1.0
                slope = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0
        else:
            # Not enough data to regress: use previous slope or default 1.0
            slope = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0

        hedge_ratio[i] = slope

        # Compute spread for this bar if prices are finite
        if np.isfinite(arr_a[i]) and np.isfinite(arr_b[i]):
            spread[i] = arr_a[i] - slope * arr_b[i]
        else:
            spread[i] = np.nan

    # Compute rolling mean and std of spread (lookback for z-score)
    s_spread = pd.Series(spread)
    roll_mean = s_spread.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use population std (ddof=0) to keep consistency and avoid NaNs with 1 sample
    roll_std = s_spread.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Calculate z-score safely (avoid division by zero / NaN)
    zscore = np.zeros(n, dtype=float)
    for i in range(n):
        if not np.isfinite(spread[i]):
            zscore[i] = 0.0
            continue
        mu = roll_mean[i]
        sigma = roll_std[i]
        if not np.isfinite(sigma) or sigma == 0.0:
            zscore[i] = 0.0
        else:
            zscore[i] = (spread[i] - mu) / sigma

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset mode.

    Returns a tuple (size, size_type, direction) or (np.nan, 0, 0) for no order.

    - size: number of units to trade (float)
    - size_type: integer code for size interpretation (we use 1 -> raw units)
    - direction: integer code for direction (1 -> long/buy, 2 -> short/sell)

    The function uses only information up to the current bar (c.i) and the
    provided arrays. It computes fixed-notional positions ($10,000 per leg) and
    scales asset B by the hedge ratio.

    To reduce order churn and avoid excessive order generation in the
    simulator, entry signals require the z-score to cross the entry threshold
    (i.e., prev_z <= threshold < curr_z) instead of simply being above it.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Protect against out-of-bounds indices
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Read current values
    curr_z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    curr_hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # If any necessary data is missing, do nothing
    if np.isnan(curr_z) or np.isnan(curr_hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Determine previous zscore for crossing detection (use only past data)
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else None

    # Current position in units for this asset (provided by wrapper in flex mode)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Module-level per-bar counter to avoid returning excessive orders for the same bar
    # (mitigates simulator behavior where last_position might not be updated between calls)
    cnt = _BAR_ORDER_COUNTER.get(i, 0)

    # Fixed notional per leg
    NOTIONAL = 10_000.0

    # Compute target unit sizes based on A's notional
    # size_a_units is always positive (magnitude)
    size_a_units = NOTIONAL / price_a if price_a != 0 else 0.0
    size_b_units = abs(curr_hr) * size_a_units

    # Map directions: 1 -> Long (buy), 2 -> Short (sell)
    DIR_LONG = 1
    DIR_SHORT = 2
    SIZE_TYPE_UNITS = 1  # interpret size as raw units

    # ENTRY CONDITIONS (threshold crossing to avoid repeated entries)
    # Only enter if there is no existing position for this asset
    if pos_now == 0.0:
        # Short Asset A, Long Asset B when zscore crosses above entry_threshold
        if curr_z > entry_threshold and (prev_z is not None and prev_z <= entry_threshold):
            if cnt < 4:  # allow a few returns per bar but guard against runaway
                _BAR_ORDER_COUNTER[i] = cnt + 1
                if col == 0:
                    # Asset A: short
                    return (size_a_units, SIZE_TYPE_UNITS, DIR_SHORT)
                else:
                    # Asset B: long
                    return (size_b_units, SIZE_TYPE_UNITS, DIR_LONG)
            return (np.nan, 0, 0)

        # Long Asset A, Short Asset B when zscore crosses below -entry_threshold
        if curr_z < -entry_threshold and (prev_z is not None and prev_z >= -entry_threshold):
            if cnt < 4:
                _BAR_ORDER_COUNTER[i] = cnt + 1
                if col == 0:
                    # Asset A: long
                    return (size_a_units, SIZE_TYPE_UNITS, DIR_LONG)
                else:
                    # Asset B: short
                    return (size_b_units, SIZE_TYPE_UNITS, DIR_SHORT)
            return (np.nan, 0, 0)

        # No entry
        return (np.nan, 0, 0)

    # EXIT CONDITIONS (only if we have an open position on this asset)
    crossed_zero = False
    if prev_z is not None:
        try:
            if prev_z * curr_z < 0:
                crossed_zero = True
        except Exception:
            crossed_zero = False

    if crossed_zero or abs(curr_z) > stop_threshold:
        # Close the position fully
        size_to_close = abs(pos_now)
        if size_to_close == 0:
            return (np.nan, 0, 0)
        if cnt < 4:
            _BAR_ORDER_COUNTER[i] = cnt + 1
            # If currently long (pos_now > 0), submit sell (short) to close
            if pos_now > 0:
                return (size_to_close, SIZE_TYPE_UNITS, DIR_SHORT)
            else:
                # If currently short (pos_now < 0), submit buy (long) to close
                return (size_to_close, SIZE_TYPE_UNITS, DIR_LONG)
        return (np.nan, 0, 0)

    # No action
    return (np.nan, 0, 0)
