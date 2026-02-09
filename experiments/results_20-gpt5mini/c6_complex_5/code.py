# Pairs trading strategy implementation for vectorbt
# Implements compute_spread_indicators and order_func as required.

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray | pd.Series,
    close_b: np.ndarray | pd.Series,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS slope) and z-score of the spread.

    Hedge ratio is computed as the slope of a linear regression of A on B
    using a rolling window of up to `hedge_lookback` periods. When not enough
    history is available (t < hedge_lookback), an expanding window is used
    (i.e., use all available data up to t). This avoids lookahead and also
    prevents NaNs at early indices.

    Spread = A - hedge_ratio * B
    Z-score = (spread - rolling_mean(spread, zscore_lookback)) /
              max(rolling_std(spread, zscore_lookback), eps)

    Args:
        close_a: 1-D price array for asset A
        close_b: 1-D price array for asset B
        hedge_lookback: lookback for rolling OLS (max window length)
        zscore_lookback: lookback for z-score mean/std

    Returns:
        dict with keys:
            - 'hedge_ratio': numpy array of same length as inputs
            - 'zscore': numpy array of same length as inputs
    """
    # Convert inputs to 1-D numpy arrays
    a = np.asarray(close_a).astype(float).ravel()
    b = np.asarray(close_b).astype(float).ravel()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling (or expanding when insufficient history) OLS slope: slope = cov(a,b)/var(b)
    for t in range(n):
        # window length: use up to hedge_lookback, but at least 2 points for slope
        window_len = min(hedge_lookback, t + 1)
        if window_len < 2:
            # not enough points to compute a slope; set to 0 for stability
            hedge_ratio[t] = 0.0
            continue

        start = t - window_len + 1
        a_win = a[start : t + 1]
        b_win = b[start : t + 1]

        mean_a = a_win.mean()
        mean_b = b_win.mean()

        denom = np.sum((b_win - mean_b) ** 2)
        if denom == 0.0:
            # Degenerate case: b constant over the window - fall back to 0
            slope = 0.0
        else:
            numer = np.sum((b_win - mean_b) * (a_win - mean_a))
            slope = numer / denom

        hedge_ratio[t] = slope

    # Compute spread
    spread = a - hedge_ratio * b

    # Rolling mean and std for z-score using up to zscore_lookback (expanding for early indices)
    zmean = np.full(n, np.nan, dtype=float)
    zstd = np.full(n, np.nan, dtype=float)
    eps = 1e-8

    for t in range(n):
        win = min(zscore_lookback, t + 1)
        start = t - win + 1
        s_win = spread[start : t + 1]
        zmean[t] = s_win.mean()
        zstd[t] = s_win.std(ddof=0)

    # Avoid division by zero in z-score
    zstd_safe = np.where(zstd < eps, eps, zstd)

    zscore = (spread - zmean) / zstd_safe

    # Ensure outputs are numpy arrays and same length as inputs
    return {"hedge_ratio": np.asarray(hedge_ratio), "zscore": np.asarray(zscore)}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset pairs trading.

    Returns a tuple (size, size_type, direction) where:
      - size: float number of units to trade (or np.nan for NoOrder)
      - size_type: integer according to vectorbt's SizeType enum
          (0 = Amount, 3 = TargetAmount). We mostly use Amount (0).
      - direction: integer according to vectorbt's Direction enum
          (0 = LongOnly, 1 = ShortOnly, 2 = Both).

    The function is designed to be called for each asset (col = 0 for A, 1 for B)
    and uses c.i (current index) and c.position_now (current position in units)
    supplied by the backtest runner.

    Logic implemented:
      - Entry when zscore > entry_threshold: short A, long B
      - Entry when zscore < -entry_threshold: long A, short B
      - Exit when zscore crosses 0 (sign change) OR |zscore| > stop_threshold
      - Position sizing: base_units = notional_per_leg / price_A
          Asset A units = ±base_units
          Asset B units = ±(hedge_ratio * base_units)

    Notes:
      - Uses past information only (zscore and hedge_ratio at index i and i-1)
      - Returns (np.nan, 0, 0) to indicate no order
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive checks and extract indicator values at i
    try:
        price_a = float(close_a[i])
        price_b = float(close_b[i])
        zs = float(zscore[i])
        hr = float(hedge_ratio[i])
    except Exception:
        return (np.nan, 0, 0)

    # If indicators or prices invalid -> no order
    if not np.isfinite(zs) or not np.isfinite(hr) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Previous zscore for crossing detection
    prev_zs = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Determine desired target positions in units for this bar
    # Base units defined by A (so B is scaled by hedge ratio)
    base_units = float(notional_per_leg / price_a)

    desired_a: float | None = None
    desired_b: float | None = None

    # Stop-loss has highest priority
    if abs(zs) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Exit when zscore crosses zero (sign change) or equals zero
        crossed_zero = False
        if i > 0 and np.isfinite(prev_zs):
            if prev_zs * zs < 0:
                crossed_zero = True
        if not crossed_zero and zs == 0.0:
            crossed_zero = True

        if crossed_zero:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if zs > entry_threshold:
                # Short A, Long B
                desired_a = -base_units
                desired_b = hr * base_units
            elif zs < -entry_threshold:
                # Long A, Short B
                desired_a = base_units
                desired_b = -hr * base_units
            else:
                # No action
                return (np.nan, 0, 0)

    # Choose the desired units for this column
    pos_now = float(getattr(c, "position_now", 0.0))
    desired = desired_a if col == 0 else desired_b

    # Defensive: if desired is None -> no order
    if desired is None:
        return (np.nan, 0, 0)

    # Compute difference (how many units to trade)
    delta = desired - pos_now

    # If no meaningful change, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Determine Direction (0=LongOnly, 1=ShortOnly, 2=Both)
    if delta > 0:
        # Want to increase in the long direction
        direction = 2 if pos_now < 0 else 0
    else:
        # delta < 0 -> want to decrease (sell) or increase short
        direction = 2 if pos_now > 0 else 1

    size = float(abs(delta))

    # size_type = 0 => Amount (number of units)
    size_type = 0

    return (size, size_type, int(direction))
