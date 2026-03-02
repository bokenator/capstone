# Pairs trading strategy utilities
# Exports:
# - compute_spread_indicators
# - order_func

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score of the spread for a pair.

    Hedge ratio is estimated with a rolling OLS slope (regressing A on B)
    using covariance/variance: slope = cov(A, B) / var(B).

    Z-score is computed from the spread = A - hedge_ratio * B using a
    rolling mean/std with `zscore_lookback` window.

    Args:
        close_a: 1D numpy array of closes for asset A
        close_b: 1D numpy array of closes for asset B
        hedge_lookback: lookback for rolling OLS (in bars)
        zscore_lookback: lookback for rolling mean/std of spread

    Returns:
        dict with keys:
            - "zscore": numpy array of z-scores (same length as inputs)
            - "hedge_ratio": numpy array of rolling hedge ratios
            (both contain NaN for warmup periods)
    """
    # Basic validation
    if close_a is None or close_b is None:
        raise ValueError("close_a and close_b must be provided")

    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Use pandas for rolling convenience
    s_a = pd.Series(a)
    s_b = pd.Series(b)

    # Rolling covariance and variance (min_periods = hedge_lookback to get full-window OLS)
    cov = s_a.rolling(window=hedge_lookback, min_periods=hedge_lookback).cov(s_b)
    var_b = s_b.rolling(window=hedge_lookback, min_periods=hedge_lookback).var()

    # Hedge ratio: slope = cov(A, B) / var(B)
    # Guard against division by zero
    hedge_ratio = cov / var_b

    # Spread
    spread = s_a - hedge_ratio * s_b

    # Rolling mean and std for the spread (z-score)
    spread_mean = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) if available; fall back to default if not
    try:
        spread_std = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)
    except TypeError:
        # Older pandas versions may not support ddof in rolling.std
        spread_std = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    # Compute zscore safely
    zscore = (spread - spread_mean) / spread_std

    # Convert to numpy arrays
    out = {
        "zscore": zscore.to_numpy(dtype=float),
        "hedge_ratio": hedge_ratio.to_numpy(dtype=float),
    }

    return out


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
    """Order function for flexible multi-asset pairs trading.

    This function is designed to be called per-column (asset) by the
    runner wrapper. It returns a simple tuple (size, size_type, direction).

    SizeType and Direction are returned as raw integer codes to be
    compatible with vectorbt's order API. The mapping used here (based on
    vectorbt's SizeType and Direction enumerations) is:

    SizeType:
        0 = Amount
        1 = Value
        2 = Percent
        3 = TargetAmount
        4 = TargetValue
        5 = TargetPercent

    Direction:
        0 = LongOnly
        1 = ShortOnly
        2 = Both

    Logic implemented:
    - Entry:
        zscore > entry_threshold  => SHORT asset A, LONG asset B
        zscore < -entry_threshold => LONG asset A, SHORT asset B
      Orders open fixed notional exposures: size_type = Value (1), size = notional_per_leg.

    - Exit:
        zscore crosses exit_threshold (typically 0) => close position for that asset
        |zscore| > stop_threshold => stop-loss, close position for that asset
      Exits are implemented as target value = 0 (size_type = TargetValue (4), size = 0).

    - When no order is to be issued, the function returns a tuple whose first
      element is np.nan. The calling wrapper treats that as NoOrder.

    Args:
        c: Order context; expects attributes `.i` (bar index) and `.col` (column index)
           and `.position_now` (current position for that asset). In flexible
           wrapper the simulation also provides these.
        close_a, close_b: numpy arrays of closes for both assets (for reference)
        zscore: numpy array of z-scores
        hedge_ratio: numpy array of hedge ratios (not used for sizing here)
        entry_threshold: threshold to enter (e.g., 2.0)
        exit_threshold: threshold to exit (e.g., 0.0)
        stop_threshold: stop-loss threshold (e.g., 3.0)
        notional_per_leg: fixed dollar exposure per leg (e.g., 10000.0)

    Returns:
        (size, size_type, direction) where size may be np.nan to indicate no order.
    """
    # Constants (raw enum integer codes)
    SIZE_TYPE_VALUE = 1
    SIZE_TYPE_TARGET_VALUE = 4
    DIRECTION_LONG = 0
    DIRECTION_SHORT = 1
    DIRECTION_BOTH = 2

    # Safely extract bar and column
    try:
        i = int(c.i)
    except Exception:
        # If context doesn't provide index, don't issue orders
        return (np.nan, 0, 0)

    # Column: 0 -> asset_a, 1 -> asset_b
    col = int(getattr(c, "col", 0))

    # Current position (number of units or value depending on context) - treat 0 as flat
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Read current and previous z-scores
    if i < 0 or i >= zscore.shape[0]:
        return (np.nan, 0, 0)

    z_now = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    z_prev = np.nan
    if i > 0:
        z_prev = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else np.nan

    # If zscore is NaN, do nothing
    if np.isnan(z_now):
        return (np.nan, 0, 0)

    # Helper to emit an exit (close) order: target value = 0
    def exit_order() -> Tuple[float, int, int]:
        return (0.0, SIZE_TYPE_TARGET_VALUE, DIRECTION_BOTH)

    # STOP-LOSS: if |z| > stop_threshold -> close
    if stop_threshold is not None and not np.isnan(stop_threshold) and abs(z_now) > float(stop_threshold):
        if pos_now != 0.0:
            return exit_order()
        else:
            return (np.nan, 0, 0)

    # EXIT on crossing the exit_threshold (e.g., crossing zero)
    crossed = False
    if not np.isnan(z_prev):
        # Crossing condition: previous on one side, now on the other side OR hitting exactly the threshold
        if (z_prev > exit_threshold and z_now <= exit_threshold) or (z_prev < exit_threshold and z_now >= exit_threshold):
            crossed = True

    if crossed and pos_now != 0.0:
        return exit_order()

    # ENTRY logic: only enter if currently flat
    if pos_now == 0.0:
        # Long/short decisions depend on column
        if z_now > entry_threshold:
            # Short Asset A, Long Asset B
            if col == 0:
                # Short asset A
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)
            else:
                # Long asset B
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)

        if z_now < -entry_threshold:
            # Long Asset A, Short Asset B
            if col == 0:
                # Long asset A
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)
            else:
                # Short asset B
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)

    # Otherwise, no order
    return (np.nan, 0, 0)
