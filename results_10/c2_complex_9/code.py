import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS) and z-score of the spread.

    Args:
        close_a: Prices for asset A (1D array-like).
        close_b: Prices for asset B (1D array-like).
        hedge_lookback: Lookback window (in bars) for rolling OLS to compute hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        A dict with keys:
            - "hedge_ratio": 1D numpy array of hedge ratios (same length as inputs).
            - "zscore": 1D numpy array of z-scores for the spread.

    Notes:
        - Hedge ratio is computed as the slope from OLS regression of A on B using
          rolling-window closed-form formula (no explicit loop per window).
        - Rolling operations produce NaNs for the warmup periods (< lookback).
    """
    # Convert to pandas Series for convenient rolling ops; preserve length
    sa = pd.Series(np.asarray(close_a, dtype=float))
    sb = pd.Series(np.asarray(close_b, dtype=float))

    n = int(hedge_lookback)
    if n < 2:
        raise ValueError("hedge_lookback must be at least 2")

    # Rolling sums
    sum_a = sa.rolling(window=n, min_periods=n).sum()
    sum_b = sb.rolling(window=n, min_periods=n).sum()
    sum_bb = (sb * sb).rolling(window=n, min_periods=n).sum()
    sum_ba = (sb * sa).rolling(window=n, min_periods=n).sum()

    # Numerator and denominator for slope (hedge ratio)
    # slope = (sum_ba - (sum_b * sum_a) / n) / (sum_bb - (sum_b * sum_b) / n)
    num = sum_ba - (sum_b * sum_a) / n
    den = sum_bb - (sum_b * sum_b) / n

    # Avoid division by zero
    hedge_ratio = num / den
    hedge_ratio = hedge_ratio.replace([np.inf, -np.inf], np.nan)

    # If denominator is zero (constant B window), set hedge ratio to NaN
    hedge_ratio[den == 0] = np.nan

    # Spread = A - hedge_ratio * B
    spread = sa - hedge_ratio * sb

    # Rolling mean and std for z-score
    zmean = spread.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).mean()
    zstd = spread.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).std(ddof=0)

    # Compute z-score safely
    zscore = (spread - zmean) / zstd
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return {
        "hedge_ratio": hedge_ratio.values.astype(float),
        "zscore": zscore.values.astype(float),
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
    """Order function for a pairs trading strategy (flexible multi-asset mode).

    This function is called once per asset (column) for the current bar. It must
    return a tuple (size, size_type, direction) or (np.nan, ..., ...) to indicate
    no order.

    Behavior:
      - Uses z-score and rolling hedge ratio computed externally.
      - Entry:
          * zscore > entry_threshold  => SHORT A, LONG B
          * zscore < -entry_threshold => LONG A, SHORT B
      - Exit:
          * zscore crosses exit_threshold (typically 0.0) => close both legs
          * |zscore| > stop_threshold => stop-loss, close both legs
      - Position sizing:
          * Base unit = notional_per_leg / price_A
          * Asset A size = +/- base_unit
          * Asset B size = +/- (base_unit * hedge_ratio)

    Notes:
      - Returns size in absolute units (size_type = Amount = 0).
      - direction uses OrderSide codes: Buy=0, Sell=1.
      - Does not use numba and returns plain Python types.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive access to arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    # Validate index
    if i < 0 or i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If indicators or prices are NaN, do nothing
    if np.isnan(price_a) or np.isnan(price_b) or np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Current position for this column (units). Use 0 if unavailable.
    pos_now = getattr(c, "position_now", None)
    if pos_now is None:
        pos_now = getattr(c, "position", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Determine signals
    is_stop = abs(z) > float(stop_threshold)

    # Crossing detection for exit through 'exit_threshold' (usually 0.0)
    crossed = False
    if i > 0:
        prev_z = float(zscore[i - 1])
        # Only consider crossing if prev_z and z are finite
        if np.isfinite(prev_z):
            # Cross when we pass through the exit_threshold
            crossed = (
                (prev_z > exit_threshold and z <= exit_threshold)
                or (prev_z < exit_threshold and z >= exit_threshold)
            )

    # Determine desired target positions (in units)
    desired_a = pos_now if col == 0 else None
    desired_b = pos_now if col == 1 else None

    # Base unit: define 1 "unit" of A as notional_per_leg / price_a (so A leg has $notional_per_leg)
    base_unit = 0.0
    if price_a > 0 and np.isfinite(price_a):
        base_unit = float(notional_per_leg) / price_a

    # Default: no new signals -> keep current positions
    # Evaluate priority: stop-loss and exit (close) take precedence over entry
    if is_stop or crossed:
        desired_a = 0.0
        desired_b = 0.0
    else:
        if z > entry_threshold:
            # SHORT A, LONG B
            desired_a = -base_unit
            desired_b = base_unit * hr
        elif z < -entry_threshold:
            # LONG A, SHORT B
            desired_a = base_unit
            desired_b = -base_unit * hr
        else:
            # No actionable signal -> no order
            # Returning np.nan signals no order for this column
            return (np.nan, 0, 0)

    # Select desired position for this column
    target = desired_a if col == 0 else desired_b

    # If target is None or NaN, do nothing
    if target is None or not np.isfinite(target):
        return (np.nan, 0, 0)

    # Compute delta (units to trade)
    delta = target - pos_now

    # If delta is effectively zero, no order
    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    size = float(abs(delta))

    # size_type: Amount (units) -> code 0
    size_type = 0

    # direction: Order side codes: Buy=0, Sell=1
    direction = 0 if delta > 0 else 1

    return (size, size_type, direction)
