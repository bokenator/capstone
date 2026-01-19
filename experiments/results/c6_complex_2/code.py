from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import vectorbt as vbt

# Internal state to avoid emitting repeated orders for the same asset in the same bar
_LAST_ORDER_BAR: int | None = None
_PLACED_IN_BAR: set = set()


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a (np.ndarray): Close prices for asset A.
        close_b (np.ndarray): Close prices for asset B.
        hedge_lookback (int): Lookback window for rolling OLS to estimate hedge ratio.
        zscore_lookback (int): Lookback window for rolling mean/std of spread.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - "hedge_ratio": numpy array of hedge ratios (slope from OLS)
            - "zscore": numpy array of z-score of the spread

    Notes:
        - Uses an expanding window when there is less than `hedge_lookback` data
          (but at least 2 points are required to compute OLS).
        - Rolling statistics are computed in a causal manner (no lookahead).
        - Any non-finite z-score values are set to 0.0 for robustness.
    """
    # Validate inputs presence
    if close_a is None or close_b is None:
        raise ValueError("close_a and close_b must be provided")

    # Convert to arrays
    arr_a = np.array(close_a, dtype=np.float64)
    arr_b = np.array(close_b, dtype=np.float64)

    # Base on length of first argument to determine output length (robust to different inputs)
    n = arr_a.shape[0]

    # If arr_b is shorter, pad with NaNs; if longer, truncate to match arr_a length
    if arr_b.shape[0] < n:
        b = np.full(n, np.nan, dtype=np.float64)
        b[: arr_b.shape[0]] = arr_b
    else:
        b = arr_b[:n].astype(np.float64)

    a = arr_a

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=np.float64)
    spread = np.full(n, np.nan, dtype=np.float64)

    # Rolling OLS for hedge ratio (slope of regressing A ~ B)
    for i in range(n):
        start = i - hedge_lookback + 1
        if start < 0:
            start = 0

        x = b[start : i + 1]
        y = a[start : i + 1]

        # Use only finite observations
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) >= 2:
            # Fully-qualified scipy call per API requirements
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = slope

            if np.isfinite(slope) and np.isfinite(a[i]) and np.isfinite(b[i]):
                spread[i] = a[i] - slope * b[i]
            else:
                spread[i] = np.nan
        else:
            hedge_ratio[i] = np.nan
            spread[i] = np.nan

    # Rolling mean and std of spread (causal)
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean()
    rolling_std = pd.Series.rolling(spread_series, window=zscore_lookback).std()

    # Compute z-score in a careful, elementwise way to avoid inf / nan
    zscore = np.full(n, np.nan, dtype=np.float64)
    rm_vals = rolling_mean.values
    rs_vals = rolling_std.values

    for i in range(n):
        if (
            np.isfinite(spread[i])
            and np.isfinite(rm_vals[i])
            and np.isfinite(rs_vals[i])
            and rs_vals[i] > 0
        ):
            zscore[i] = (spread[i] - rm_vals[i]) / rs_vals[i]
        else:
            zscore[i] = np.nan

    # Replace remaining non-finite z-scores with 0.0 to ensure no NaNs after warmup
    zscore = np.where(np.isfinite(zscore), zscore, 0.0)

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


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
    Order function for pairs trading strategy.

    This function computes the desired position (in units) for the asset corresponding
    to c.col and returns an order tuple (size, size_type, direction). The wrapper
    provided by the backtest harness will convert this tuple into an actual Order.

    Position sizing:
        - Fixed notional per base leg: $10,000
        - Asset A units = notional / price_a
        - Asset B units = hedge_ratio * asset_a_units (opposite sign)

    Signals:
        - Enter short A / long B when zscore > entry_threshold
        - Enter long A / short B when zscore < -entry_threshold
        - Exit when zscore crosses zero (sign change) or |zscore| > stop_threshold

    Returns:
        Tuple of (size, size_type, direction):
            - size (float): absolute size (in units) to trade
            - size_type (int): 0 => absolute size (units)
            - direction (int): 1 => buy (increase), -1 => sell (decrease)

        If no order is required, returns a tuple with size = np.nan which the wrapper
        will ignore.
    """
    global _LAST_ORDER_BAR, _PLACED_IN_BAR

    i = int(getattr(c, "i"))
    col = int(getattr(c, "col", 0))

    # Reset per-bar set when we move to a new bar
    if _LAST_ORDER_BAR != i:
        _LAST_ORDER_BAR = i
        _PLACED_IN_BAR.clear()

    # If we already placed an order for this column in this bar, do nothing
    if col in _PLACED_IN_BAR:
        return (np.nan, 0, 0)

    # Current position for this asset (in units). Provided by wrapper's simulated context.
    position_now = float(getattr(c, "position_now", 0.0))

    # Basic validation
    if i < 0 or i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    z = float(zscore[i]) if np.isfinite(zscore[i]) else 0.0
    h = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # If hedge ratio or prices are not available, do nothing
    if not np.isfinite(h) or not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine previous z to detect zero crossing
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else 0.0

    # Fixed notional per base leg (asset A)
    fixed_notional = 10_000.0
    base_units = fixed_notional / price_a

    # Default: hold current position
    desired_a = position_now if col == 0 else None  # placeholder
    desired_b = None

    # Stop-loss: close both legs if |z| exceeds stop_threshold
    if np.abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0

    # Exit on mean reversion (crossing zero) or small z (<= exit_threshold)
    elif prev_z * z < 0 or np.abs(z) <= exit_threshold:
        desired_a = 0.0
        desired_b = 0.0

    # Entry conditions
    elif z > entry_threshold:
        # Short A, long B
        desired_a = -base_units
        desired_b = +h * base_units

    elif z < -entry_threshold:
        # Long A, short B
        desired_a = +base_units
        desired_b = -h * base_units

    else:
        # No signal: do nothing
        return (np.nan, 0, 0)

    # Determine target for this column
    target = desired_a if col == 0 else desired_b

    # Compute delta from current position
    delta = target - position_now

    # If the delta is effectively zero, do nothing (tolerance to avoid tiny repeated orders)
    if not np.isfinite(delta) or np.abs(delta) < 1e-6:
        return (np.nan, 0, 0)

    size = float(np.abs(delta))
    direction = 1 if delta > 0 else -1

    # Mark that we've placed an order for this column in this bar
    _PLACED_IN_BAR.add(col)

    # size_type = 0 indicates absolute size (units)
    size_type = 0

    return (size, int(size_type), int(direction))
