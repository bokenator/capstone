"""
Pairs trading strategy helpers for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20)
    Computes rolling hedge ratio (OLS), spread, and z-score.

- order_func(c, close_a, close_b, zscore, hedge_ratio,
             entry_threshold, exit_threshold, stop_threshold, notional_per_leg)
    Order function compatible with the provided flexible multi-asset wrapper.

Notes:
- No numba is used.
- Returns plain Python tuples from order_func.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A as 1D numpy array.
        close_b: Prices for asset B as 1D numpy array.
        hedge_lookback: Lookback (in bars) for rolling OLS regression to estimate hedge ratio.
        zscore_lookback: Lookback (in bars) to compute rolling mean/std of spread for z-score.

    Returns:
        Dict with keys:
            - "hedge_ratio": numpy array of hedge ratios (same length as inputs)
            - "spread": numpy array of spread values (Price_A - hedge_ratio * Price_B)
            - "zscore": numpy array of z-scores

    Notes on implementation:
        - Hedge ratio at time t is computed by regressing Price_A on Price_B over the
          previous `hedge_lookback` bars (requires full window of non-NaN values).
        - Z-score uses population std (ddof=0) on the spread with `zscore_lookback` window.
        - Warm-up periods produce NaNs.
    """
    # Basic validation
    if close_a is None or close_b is None:
        raise ValueError("close_a and close_b must be provided")

    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = close_a.shape[0]

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio: Price_A = beta * Price_B + alpha
    # We require a full window of finite values to compute the regression.
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be positive")

    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        a_win = close_a[start_idx : end_idx + 1]
        b_win = close_b[start_idx : end_idx + 1]

        # Ensure full window is finite
        if not (np.isfinite(a_win).all() and np.isfinite(b_win).all()):
            # leave hedge_ratio[end_idx] as NaN
            continue

        # Design matrix [B, 1] to solve for [beta, alpha]
        X = np.vstack([b_win, np.ones_like(b_win)]).T
        try:
            # Solve least squares; params[0] is beta (slope)
            params, *_ = np.linalg.lstsq(X, a_win, rcond=None)
            beta = float(params[0])
            hedge_ratio[end_idx] = beta
        except np.linalg.LinAlgError:
            # Numerically unstable window; keep NaN
            continue

    # Compute spread: Price_A - hedge_ratio * Price_B
    # Broadcast will produce NaNs where hedge_ratio is NaN
    spread = close_a - hedge_ratio * close_b

    # Compute rolling mean and std of spread for z-score
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be positive")

    s = pd.Series(spread)
    rolling_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to be consistent and avoid NaNs when variance is zero
    rolling_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = np.isfinite(spread) & np.isfinite(rolling_mean) & np.isfinite(rolling_std) & (rolling_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]

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
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset mode.

    This function follows the specified pairs trading logic:
      - Entry: |zscore| > entry_threshold -> open pair (short A / long B or vice versa)
      - Exit: zscore crosses exit_threshold (typically 0.0) -> close both legs
      - Stop-loss: |zscore| > stop_threshold -> close both legs
      - Position sizing: scale base units by notional_per_leg for asset A
        (units_A = notional_per_leg / price_A), and set units_B = units_A * hedge_ratio.

    Returns a tuple (size, size_type, direction) where:
      - size: number of units to trade (float)
      - size_type: 0 (absolute quantity)
      - direction: 1 for buy, 2 for sell

    Notes:
      - The wrapper expects a tuple of at least length 3. Returning (np.nan, 0, 0)
        indicates no order for this asset at this bar.
      - We avoid using vectorbt enums or numba constructs.
    """
    # Simple helpers / constants
    BUY = 1
    SELL = 2

    def no_order():
        return (float('nan'), 0, 0)

    # Basic indexing and safety
    i = getattr(c, "i", None)
    col = getattr(c, "col", None)
    if i is None or col is None:
        # Cannot proceed without index/column information
        return no_order()

    # Ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i < 0 or i >= n:
        return no_order()

    curr_z = zscore[i]
    curr_hr = hedge_ratio[i]

    # Determine current position for this column
    # c may provide position_now (float) or last_position (array-like)
    curr_pos = 0.0
    if hasattr(c, "position_now"):
        try:
            curr_pos = float(getattr(c, "position_now"))
        except Exception:
            # Fall back to other attribute
            curr_pos = 0.0
    elif hasattr(c, "last_position"):
        try:
            last_pos = getattr(c, "last_position")
            curr_pos = float(last_pos[col])
        except Exception:
            curr_pos = 0.0

    # Get prices
    price = close_a[i] if col == 0 else close_b[i]
    price_a = close_a[i]

    # Validate key numbers
    if not np.isfinite(price) or not np.isfinite(price_a) or not np.isfinite(curr_z):
        return no_order()

    # Stop-loss: absolute z-score beyond stop_threshold -> close any open position
    if stop_threshold is not None and np.isfinite(stop_threshold) and abs(curr_z) > stop_threshold:
        if curr_pos == 0 or not np.isfinite(curr_pos):
            return no_order()
        # To close: if currently long (>0) then sell, if short (<0) then buy
        if curr_pos > 0:
            return (float(abs(curr_pos)), 0, SELL)
        else:
            return (float(abs(curr_pos)), 0, BUY)

    # Exit condition: z-score crosses exit_threshold (typically 0.0) OR abs(curr_z) <= exit_threshold
    crossed = False
    if i > 0 and np.isfinite(zscore[i - 1]) and np.isfinite(curr_z):
        prev_z = zscore[i - 1]
        # Cross through the threshold from either side
        if (prev_z > exit_threshold and curr_z <= exit_threshold) or (prev_z < exit_threshold and curr_z >= exit_threshold):
            crossed = True
    # Also close if we're already within the exit band
    if np.isfinite(curr_z) and abs(curr_z) <= exit_threshold:
        crossed = True

    if crossed:
        if curr_pos == 0 or not np.isfinite(curr_pos):
            return no_order()
        if curr_pos > 0:
            return (float(abs(curr_pos)), 0, SELL)
        else:
            return (float(abs(curr_pos)), 0, BUY)

    # Entry conditions
    desired_a_sign = 0
    if np.isfinite(curr_z):
        if curr_z > entry_threshold:
            # zscore high -> short A, long B
            desired_a_sign = -1
        elif curr_z < -entry_threshold:
            # zscore low -> long A, short B
            desired_a_sign = 1

    # If no desired position (between thresholds), do nothing
    if desired_a_sign == 0:
        return no_order()

    # Determine desired sign for this column
    if col == 0:
        desired_sign = desired_a_sign
    else:
        # desired sign for B takes hedge ratio into account:
        # desired_B_sign = - desired_A_sign * sign(hedge_ratio)
        if not np.isfinite(curr_hr) or curr_hr == 0:
            # Cannot determine hedge exposure; skip trading B
            return no_order()
        hr_sign = 1 if curr_hr > 0 else -1
        desired_sign = -desired_a_sign * hr_sign

    # Compute desired absolute size for this column
    # We scale the pair by A's notional: units_A = notional_per_leg / price_A
    if price_a == 0 or not np.isfinite(price_a):
        return no_order()

    units_a = float(notional_per_leg) / float(price_a)

    if col == 0:
        desired_abs_size = units_a
    else:
        desired_abs_size = abs(units_a * float(curr_hr))

    # No meaningful trade if desired size is zero or not finite
    if not np.isfinite(desired_abs_size) or desired_abs_size <= 0:
        return no_order()

    # Determine trade size to reach desired position from current position
    # If current position has opposite sign, we trade the sum of absolute sizes (close + open)
    trade_size = 0.0
    if curr_pos == 0 or not np.isfinite(curr_pos):
        trade_size = desired_abs_size
    else:
        # Same sign
        if curr_pos * desired_sign > 0:
            # If we already have a position in the desired direction, only adjust incrementally
            diff = desired_abs_size - abs(curr_pos)
            if diff <= 0:
                return no_order()
            trade_size = float(diff)
        else:
            # Opposite sign: close existing position and open desired size in one trade
            trade_size = float(abs(curr_pos) + desired_abs_size)

    if trade_size <= 0 or not np.isfinite(trade_size):
        return no_order()

    direction = BUY if desired_sign > 0 else SELL

    return (float(trade_size), 0, int(direction))
