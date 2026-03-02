from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: 1D array of prices for asset A
        close_b: 1D array of prices for asset B
        hedge_lookback: lookback window for rolling OLS (in periods)
        zscore_lookback: lookback window for rolling mean/std of spread

    Returns:
        Dictionary containing at least:
            - "hedge_ratio": np.ndarray of rolling hedge ratios (slope)
            - "zscore": np.ndarray of z-score of the spread
            - "spread": np.ndarray of computed spread (Price_A - hr * Price_B)
            - "rolling_mean": np.ndarray of rolling mean of spread
            - "rolling_std": np.ndarray of rolling std of spread

    Notes:
        - Uses scipy.stats.linregress on each rolling window to compute the slope
        - Aligns outputs with input arrays; initial entries where there is
          insufficient data are NaN
    """
    # Input validation and normalization
    a = np.asarray(close_a, dtype=float).ravel()
    b = np.asarray(close_b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Rolling OLS to get hedge ratio (slope of Price_A ~ Price_B)
    # We iterate windows and compute slope with linregress; handle NaNs
    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        y = a[start : i + 1]
        x = b[start : i + 1]

        # Exclude rows with NaNs in either series
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            # If regression fails for numerical reasons, keep NaN
            hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    valid = ~np.isnan(hedge_ratio)
    spread[valid] = a[valid] - hedge_ratio[valid] * b[valid]

    # Rolling mean and std for z-score using pandas (handles NaNs nicely)
    s = pd.Series(spread)
    rolling_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) for z-score consistency; adjust if desired
    rolling_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(rolling_mean)) & (~np.isnan(rolling_std)) & (rolling_std != 0)
    zscore[valid_z] = (spread[valid_z] - rolling_mean[valid_z]) / rolling_std[valid_z]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading strategy in flexible multi-asset mode.

    This function is called separately for each asset (column). It returns a
    tuple (size, size_type, direction) where:
        - size: float (interpreted according to size_type)
        - size_type: int (we use 'notional' sizing encoded as 2)
        - direction: int (1 for long/buy, 2 for short/sell)

    Important notes about encoding:
        - To avoid importing vectorbt enums (which may be restricted in the
          testing environment), integer codes are used directly. These map to
          common internal codes in vectorbt: 2 => notional sizing, 1 => long,
          2 => short. The NoOrder sentinel is produced by returning size=NaN.

    Trading logic (per specification):
        - Entry when zscore > entry_threshold: SHORT A, LONG B (hedge_ratio units)
        - Entry when zscore < -entry_threshold: LONG A, SHORT B (hedge_ratio units)
        - Exit when zscore crosses exit_threshold (usually 0.0) OR abs(zscore) > stop_threshold
        - Position sizing: scale base 1 unit of A and hedge_ratio units of B so
          that A has notional ~= notional_per_leg; B's notional is computed to
          preserve the hedge ratio in units.

    Args:
        c: order context with attributes i (bar index), col (column index), and
           position_now (current position in units, positive for long,
           negative for short). In flexible mode the runner provides a
           simulated context.
        close_a, close_b: arrays of close prices
        zscore, hedge_ratio: arrays computed by compute_spread_indicators
        entry_threshold, exit_threshold, stop_threshold: thresholds
        notional_per_leg: fixed notional per leg (e.g., 10000.0)

    Returns:
        (size, size_type, direction) tuple. If no order, return (np.nan, 0, 0).
    """
    # Encoding (avoid importing vectorbt enums directly)
    SIZE_TYPE_NOTIONAL = 2
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    # Safely extract context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Normalize position_now to a float (units). If unavailable, assume 0.
    pos_now_raw = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now_raw)
    except Exception:
        # If position_now is an array-like, try to sum it
        try:
            pos_now = float(np.sum(np.asarray(pos_now_raw, dtype=float)))
        except Exception:
            pos_now = 0.0

    # Bounds checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # Require valid indicators to act
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Exit / stop conditions
    close_signal = False
    # Stop-loss has highest priority
    if not np.isnan(stop_threshold) and abs(z) > float(stop_threshold):
        close_signal = True
    else:
        # Detect crossing of exit_threshold (usually 0.0)
        if i > 0:
            prev_z = zscore[i - 1]
            if not np.isnan(prev_z):
                # Closed when z crosses the exit threshold from either side
                # e.g., prev_z > exit_threshold and z <= exit_threshold
                if (prev_z > exit_threshold and z <= exit_threshold) or (
                    prev_z < -exit_threshold and z >= -exit_threshold
                ):
                    close_signal = True

    if close_signal:
        # If we have no position, nothing to close
        if pos_now == 0:
            return (np.nan, 0, 0)

        # Compute notional required to close current position
        current_price = price_a if col == 0 else price_b
        size_notional = abs(pos_now) * current_price
        # Direction to close: if currently long (pos>0) -> short/sell; if short -> buy/long
        close_direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
        return (size_notional, SIZE_TYPE_NOTIONAL, close_direction)

    # Entry conditions
    # Scale base 1 unit of A to notional_per_leg, then compute B units = hedge_ratio * scaled_A
    scale_a = notional_per_leg / price_a if price_a > 0 else 0.0
    units_b = hr * scale_a
    notional_b = abs(units_b) * price_b

    # Long/Short entries
    if z > entry_threshold:
        # SHORT A, LONG B
        if pos_now == 0:
            if col == 0:
                # Short A: use notional sizing on A leg
                return (float(notional_per_leg), SIZE_TYPE_NOTIONAL, DIRECTION_SHORT)
            else:
                # Long B: sized to preserve hedge ratio in units
                # If computed notional_b is zero (due to price issues), skip
                if notional_b <= 0:
                    return (np.nan, 0, 0)
                return (float(notional_b), SIZE_TYPE_NOTIONAL, DIRECTION_LONG)

    if z < -entry_threshold:
        # LONG A, SHORT B
        if pos_now == 0:
            if col == 0:
                return (float(notional_per_leg), SIZE_TYPE_NOTIONAL, DIRECTION_LONG)
            else:
                if notional_b <= 0:
                    return (np.nan, 0, 0)
                return (float(notional_b), SIZE_TYPE_NOTIONAL, DIRECTION_SHORT)

    # No action
    return (np.nan, 0, 0)
