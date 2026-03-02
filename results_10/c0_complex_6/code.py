import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

# Constants used for order tuple mapping
# These integer values map to vectorbt's internal enums when passed into order_nb.
# We avoid importing vbt.portfolio.enums as instructed. The mapping below is the
# conventional mapping used by vectorbt: SizeType.Amount == 0, Direction.Long == 1, Direction.Short == 2.
SIZE_TYPE_AMOUNT = 0
DIRECTION_LONG = 1
DIRECTION_SHORT = 2


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling OLS hedge ratio and z-score of the spread between two assets.

    Args:
        close_a: Prices of asset A as 1D numpy array.
        close_b: Prices of asset B as 1D numpy array.
        hedge_lookback: Lookback window (in bars) for rolling OLS to estimate hedge ratio.
        zscore_lookback: Lookback window for rolling mean/std of the spread to compute z-score.

    Returns:
        A dictionary with numpy arrays (same length as inputs):
            - 'hedge_ratio': rolling hedge ratio (slope) from regressing A on B
            - 'spread': spread series defined as A - hedge_ratio * B
            - 'spread_mean': rolling mean of spread
            - 'spread_std': rolling std of spread (population std, ddof=0)
            - 'zscore': z-score of spread

    Notes:
        - Handles NaNs and warm-up periods by returning NaN where insufficient data exists.
        - Uses a simple OLS slope estimator per rolling window: slope = cov(X, Y) / var(X)
    """
    # Validate and prepare inputs
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if len(close_a) != len(close_b):
        raise ValueError("close_a and close_b must have the same length")
    if hedge_lookback <= 1:
        raise ValueError("hedge_lookback must be > 1")
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    n = len(close_a)

    # Rolling hedge ratio (slope) via OLS per window
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling slope for windows where we have enough data
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        x = close_b[start_idx : end_idx + 1]
        y = close_a[start_idx : end_idx + 1]

        # If any NaNs in the window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[end_idx] = np.nan
            continue

        mean_x = x.mean()
        mean_y = y.mean()

        denom = np.sum((x - mean_x) ** 2)
        if denom == 0 or np.isnan(denom):
            hedge_ratio[end_idx] = np.nan
            continue

        num = np.sum((x - mean_x) * (y - mean_y))
        hedge_ratio[end_idx] = num / denom

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    if valid_hr.any():
        spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std of spread using pandas (handles NaNs/warmup elegantly)
    s = pd.Series(spread)
    spread_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to be consistent and avoid small-sample bias
    spread_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score where possible
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std != 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
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
    Order function for flexible multi-asset pairs trading.

    This function is called for each asset (column) on each bar. It returns a tuple
    (size, size_type, direction) that will be converted into an order by the runner.

    Logic implemented:
      - Uses pair z-score and rolling hedge ratio provided as arrays.
      - Entry:
          z > entry_threshold  -> Short A, Long B (A: -1 unit, B: +hedge_ratio units)
          z < -entry_threshold -> Long A, Short B (A: +1 unit, B: -hedge_ratio units)
        Units are scaled by notional_per_leg for asset A (base leg). Asset B units follow via hedge_ratio.
      - Exit:
          z crosses 0.0 -> close positions
          |z| > stop_threshold -> stop-loss, close positions
      - Position sizing: fixed notional_per_leg for base leg (A); B scaled by hedge_ratio.

    Important:
      - The function must return (np.nan, int, int) to indicate no order (the wrapper treats NaN size as NoOrder).
      - We return integer codes mapped to vectorbt enums: SIZE_TYPE_AMOUNT=0, DIRECTION_LONG=1, DIRECTION_SHORT=2.

    Args:
        c: Order context-like object. Expected attributes used: i (bar index), col (column index), position_now (current position for the column).
        close_a, close_b: price arrays
        zscore, hedge_ratio: indicator arrays
        entry_threshold, exit_threshold, stop_threshold: thresholds
        notional_per_leg: fixed dollar amount per base leg

    Returns:
        Tuple (size, size_type, direction)
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Current position for this asset (number of units, signed)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic bounds/safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else float("nan")
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else float("nan")

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If indicators or prices are NaN, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # Base number of units for asset A to achieve fixed notional per leg
    base_units_a = notional_per_leg / price_a

    # Determine target shares for this bar depending on z
    target_shares = 0.0
    if z > entry_threshold:
        # Short A, Long B (A: -1 * base_units_a, B: +hr * base_units_a)
        target_shares = -base_units_a if col == 0 else (hr * base_units_a)
    elif z < -entry_threshold:
        # Long A, Short B (A: +1 * base_units_a, B: -hr * base_units_a)
        target_shares = base_units_a if col == 0 else (-hr * base_units_a)
    else:
        # No new entry signal; check exits below
        target_shares = None

    # Exit conditions (close positions) - stop-loss or z cross 0
    # Stop-loss: absolute z exceeds threshold
    if abs(z) > stop_threshold:
        # Close any open position on this asset
        if abs(pos_now) < 1e-12:
            return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
        # To close: if currently long (pos_now>0) -> submit SELL (SHORT) of abs(pos_now)
        if pos_now > 0:
            return (abs(pos_now), SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
        else:
            # currently short -> submit BUY (LONG) to cover
            return (abs(pos_now), SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # Zero-cross exit detection: if previous z exists and sign changed
    if i > 0:
        prev_z = zscore[i - 1]
        if not np.isnan(prev_z) and not np.isnan(z) and prev_z * z < 0:
            if abs(pos_now) < 1e-12:
                return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            if pos_now > 0:
                return (abs(pos_now), SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
            else:
                return (abs(pos_now), SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # If we have an explicit target_shares (entry), compute order to reach target
    if target_shares is not None:
        # desired_sign: +1 for long, -1 for short, 0 for flat
        if abs(target_shares) < 1e-12:
            desired_sign = 0
        else:
            desired_sign = 1 if target_shares > 0 else -1

        target_abs = abs(target_shares)
        current_abs = abs(pos_now)

        # If no position and target is flat -> nothing
        if desired_sign == 0 and current_abs < 1e-12:
            return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

        # Determine direction to submit for this asset
        submit_direction = DIRECTION_LONG if desired_sign > 0 else DIRECTION_SHORT

        # If currently on the same side (same sign)
        if (pos_now > 0 and desired_sign > 0) or (pos_now < 0 and desired_sign < 0):
            # If already at or above target magnitude, do nothing
            if current_abs >= target_abs - 1e-12:
                return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            # Otherwise top-up by the difference
            size_to_submit = target_abs - current_abs
            if size_to_submit <= 0:
                return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            return (float(size_to_submit), SIZE_TYPE_AMOUNT, submit_direction)

        # If currently flat (no position)
        if abs(pos_now) < 1e-12:
            if target_abs <= 0:
                return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            return (float(target_abs), SIZE_TYPE_AMOUNT, submit_direction)

        # If currently on the opposite side, submit enough to flip and reach target
        # e.g., if currently long P and target is short T -> submit sell of (P + T)
        size_to_submit = current_abs + target_abs
        if size_to_submit <= 0:
            return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
        return (float(size_to_submit), SIZE_TYPE_AMOUNT, submit_direction)

    # No action
    return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
