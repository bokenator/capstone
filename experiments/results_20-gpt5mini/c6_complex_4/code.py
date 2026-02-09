import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio, spread and z-score for a pairs strategy.

    Hedge ratio is computed with a rolling OLS (regression of A on B) using a
    lookback window of `hedge_lookback`. For early bars where the full
    lookback isn't available, an expanding-window regression is used (i.e., use
    all available past data up to the current bar).

    Z-score is computed as (spread - rolling_mean) / rolling_std with a
    lookback of `zscore_lookback`. Small numerical eps is added to the
    denominator to avoid division by zero. Any remaining NaNs in the z-score
    (very early bars) are filled with 0.0 to ensure deterministic output.

    Args:
        close_a: 1-D array of prices for asset A.
        close_b: 1-D array of prices for asset B.
        hedge_lookback: Lookback (in bars) for rolling OLS hedge ratio.
        zscore_lookback: Lookback for rolling mean/std of the spread.

    Returns:
        Dict with keys:
            - "zscore": np.ndarray, same length as inputs
            - "hedge_ratio": np.ndarray, same length as inputs
            - "spread": np.ndarray, same length as inputs
    """
    # Convert inputs to 1-D numpy arrays
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = len(a)

    # Prepare hedge ratio array
    hedge_ratio = np.zeros(n, dtype=float)

    # Use a fallback previous hedge ratio to fill any degenerate windows
    prev_hr = 1.0

    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = b[start : i + 1]
        y = a[start : i + 1]

        # Require at least 2 points to compute a slope
        if x.size >= 2 and not np.all(np.isnan(x)) and not np.all(np.isnan(y)):
            # If x has zero variance, fallback to previous hedge ratio
            if np.nanvar(x) == 0:
                slope = prev_hr
            else:
                # OLS slope (regress A on B): A = alpha + beta * B => hedge_ratio = beta
                # Use np.polyfit which is robust enough here.
                try:
                    slope = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    slope = prev_hr

            if np.isfinite(slope):
                prev_hr = slope
        else:
            slope = prev_hr

        hedge_ratio[i] = slope

    # Compute spread using the hedge ratio at each time (uses only past/current data)
    spread = a - hedge_ratio * b

    # Rolling z-score (use pandas for convenience)
    spread_s = pd.Series(spread)
    # Use min_periods=2 to avoid degenerate single-point std=0; fill early NaNs with 0
    roll_mean = spread_s.rolling(window=zscore_lookback, min_periods=2).mean()
    roll_std = spread_s.rolling(window=zscore_lookback, min_periods=2).std(ddof=0)

    eps = 1e-8
    zscore = ((spread_s - roll_mean) / (roll_std + eps)).fillna(0.0).to_numpy()

    return {
        "zscore": zscore,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
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
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """Order function for a pairs trading strategy.

    This is written for the flexible multi-asset mode where the function will be
    called for each column (asset) at each bar. The function returns a tuple
    (size, size_type, direction) understood by vectorbt's order engine.

    Rules implemented:
    - Entry when zscore > entry_threshold or zscore < -entry_threshold
      * z > entry_threshold: SHORT A, LONG B (B scaled by |hedge_ratio|)
      * z < -entry_threshold: LONG A, SHORT B
    - Exit when zscore crosses zero (sign change) or when |zscore| > stop_threshold
    - Position sizing: compute units for asset A as notional_per_leg / price_a
      and scale asset B units by abs(hedge_ratio). This makes position_b ~=
      hedge_ratio * position_a in units (sign handled separately).

    Notes:
    - Uses SizeType Amount for unit-based orders.
    - To close positions we issue an offsetting Amount order equal to the
      current position magnitude (using c.position_now provided by the context),
      which is robust and explicit.

    Args:
        c: Order context provided by vectorbt (or the test harness wrapper).
        close_a: np.ndarray of asset A close prices.
        close_b: np.ndarray of asset B close prices.
        zscore: np.ndarray of z-score values (aligned with closes).
        hedge_ratio: np.ndarray of hedge ratios (aligned with closes).
        entry_threshold: Threshold to enter (default 2.0).
        exit_threshold: Threshold to exit (unused directly; exit on zero-cross).
        stop_threshold: Stop-loss threshold (default 3.0).
        notional_per_leg: Notional per leg (dollars).

    Returns:
        (size, size_type, direction) tuple. Return (np.nan, 0, 0) to indicate
        no order for this column on this bar.
    """
    # Constants from vectorbt enums (use integer values documented in vbt):
    SIZE_TYPE_AMOUNT = 0  # Amount (units)
    SIZE_TYPE_VALUE = 1   # Value ($)
    SIZE_TYPE_TARGET_AMOUNT = 3  # Target amount (units)

    DIR_LONG_ONLY = 0
    DIR_SHORT_ONLY = 1
    DIR_BOTH = 2

    # Guard and extract index/column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Basic bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Read indicators and prices at current bar
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current position (units) for this column (provided by wrapper/context)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # If indicator is NaN, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    eps = 1e-8

    # Compute base units for asset A and scaled units for asset B
    size_a_units = (notional_per_leg / price_a) if price_a > 0 else 0.0
    size_b_units = abs(hr) * size_a_units

    # Exit conditions: zero-crossing or stop-loss
    exit_condition = False
    if i > 0:
        z_prev = float(zscore[i - 1])
        # Zero crossing: sign change between previous and current (exclude flat early zeros)
        if (z_prev * z) < 0:
            exit_condition = True
    if abs(z) > stop_threshold:
        exit_condition = True

    if exit_condition:
        # If we currently have a position, issue an offsetting Amount order to close it
        if abs(pos_now) > eps:
            # Use direction BOTH for offsetting/closing operations so that the order
            # can reduce/flip any sign of position as needed.
            return (abs(pos_now), SIZE_TYPE_AMOUNT, DIR_BOTH)
        else:
            return (np.nan, 0, 0)

    # Entry conditions
    if abs(z) > entry_threshold:
        # Only enter if currently flat
        if abs(pos_now) < eps:
            if z > 0:
                # SHORT A, LONG B (B direction depends on sign of hedge ratio)
                if col == 0:
                    # Short asset A with units sized by notional
                    return (size_a_units, SIZE_TYPE_AMOUNT, DIR_SHORT_ONLY)
                else:
                    if size_b_units <= eps:
                        return (np.nan, 0, 0)
                    direction_b = DIR_LONG_ONLY if hr >= 0 else DIR_SHORT_ONLY
                    return (size_b_units, SIZE_TYPE_AMOUNT, direction_b)
            else:
                # z < 0: LONG A, SHORT B
                if col == 0:
                    return (size_a_units, SIZE_TYPE_AMOUNT, DIR_LONG_ONLY)
                else:
                    if size_b_units <= eps:
                        return (np.nan, 0, 0)
                    direction_b = DIR_SHORT_ONLY if hr >= 0 else DIR_LONG_ONLY
                    return (size_b_units, SIZE_TYPE_AMOUNT, direction_b)

    # Otherwise, no order
    return (np.nan, 0, 0)
