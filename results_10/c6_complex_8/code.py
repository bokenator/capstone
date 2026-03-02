# Pairs trading strategy implementation for vectorbt
# Exports: compute_spread_indicators, order_func

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def _to_1d_array(x: Any) -> np.ndarray:
    """
    Convert input to a 1D numpy float array in a robust way.
    Handles pandas Series/DataFrame and numpy arrays of various shapes.
    If DataFrame or 2D array is provided, the first column is taken.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0].to_numpy(dtype=float)
        return x.values.ravel().astype(float)

    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)

    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.array([float(arr)])
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # Take first column if 2D
        if arr.shape[1] >= 1:
            return arr[:, 0].astype(float)
        return arr.ravel().astype(float)

    # For higher dimensions, flatten
    return arr.ravel().astype(float)


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread.

    - Hedge ratio: rolling OLS regression coefficient (regress A on B).
      Uses a variable window for the first bars (uses up to t+1 samples when t+1 < hedge_lookback)
      to avoid NaNs and lookahead.

    - Z-score: (spread - rolling_mean) / rolling_std computed with lookback zscore_lookback.
      Uses variable window at the beginning as well.

    Args:
        close_a: Prices of asset A (array-like or pandas Series/DataFrame).
        close_b: Prices of asset B (array-like or pandas Series/DataFrame).
        hedge_lookback: Lookback window for rolling OLS (default 60).
        zscore_lookback: Lookback window for rolling mean/std of spread (default 20).

    Returns:
        Dict with keys:
          - "hedge_ratio": np.ndarray of hedge ratios (length = len(close_a))
          - "zscore": np.ndarray of z-score values (length = len(close_a))

    Notes:
        - No lookahead: each value at index t is computed using data up to index t only.
        - Avoids returning NaNs by using available history when full lookback not available.
    """
    # Convert inputs to 1D numpy arrays
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)

    # Small epsilon to avoid division by zero
    EPS = 1e-8

    # Compute rolling OLS beta (regress A on B): beta = cov(A,B)/var(B)
    for t in range(n):
        start = 0 if t + 1 <= hedge_lookback else t + 1 - hedge_lookback
        A_win = a[start : t + 1]
        B_win = b[start : t + 1]

        if B_win.size == 0:
            hedge_ratio[t] = hedge_ratio[t - 1] if t > 0 else 0.0
            continue

        mean_A = A_win.mean()
        mean_B = B_win.mean()

        denom = ((B_win - mean_B) ** 2).sum()
        if denom <= EPS:
            # If variance of B is zero (e.g., constant price), keep previous hedge ratio or 0
            hedge_ratio[t] = hedge_ratio[t - 1] if t > 0 else 0.0
        else:
            num = ((B_win - mean_B) * (A_win - mean_A)).sum()
            hedge_ratio[t] = float(num / denom)

    # Compute spread using the hedge ratio at each time (no lookahead)
    spread = a - hedge_ratio * b

    # Compute rolling mean/std for spread and z-score
    zscore = np.zeros(n, dtype=float)
    for t in range(n):
        start = 0 if t + 1 <= zscore_lookback else t + 1 - zscore_lookback
        s_win = spread[start : t + 1]
        if s_win.size == 0:
            zscore[t] = 0.0
            continue
        mu = float(s_win.mean())
        sigma = float(s_win.std(ddof=0))
        if sigma <= EPS:
            zscore[t] = 0.0
        else:
            zscore[t] = float((spread[t] - mu) / sigma)

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
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset mode (pairs trading).

    Returns a tuple (size, size_type, direction).
    - size: the order size (Amount = number of units)
    - size_type: integer enum from vbt.portfolio.enums.SizeType (we use Amount)
    - direction: integer enum from vbt.portfolio.enums.OrderSide (Buy=0, Sell=1)

    Logic (from strategy description):
    - Entry: when zscore > entry_threshold => Short A, Long B
             when zscore < -entry_threshold => Long A, Short B
    - Exit: when zscore crosses 0 => close both; when |zscore| > stop_threshold => stop loss & close

    Position sizing:
    - We use fixed notional_per_leg to size asset A in USD, convert to units, then scale asset B
      in units by hedge_ratio so that units_B = abs(hedge_ratio) * units_A. This enforces the
      hedge_ratio relationship between instrument units (position_b ~= -hedge_ratio * position_a).

    Notes:
    - The function is called once per asset (column) per bar. It must be stateless; current
      position is read from c.position_now and decisions must only use information up to c.i.
    - Return (np.nan, 0, 0) to indicate no order.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive indexing
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, int(vbt.portfolio.nb.SizeType.Amount), int(vbt.portfolio.nb.OrderSide.Buy))

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0

    # Determine current position (units). position_now may be a scalar or array-like
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        # If it's an array or something else, be defensive
        try:
            arr_pos = np.asarray(pos_now)
            if arr_pos.size == 1:
                pos_now = float(arr_pos.item())
            else:
                pos_now = float(arr_pos[0])
        except Exception:
            pos_now = 0.0

    # Use Amount sizing (units)
    SIZE_TYPE_AMOUNT = int(vbt.portfolio.nb.SizeType.Amount)  # 0
    ORDER_SIDE_BUY = int(vbt.portfolio.nb.OrderSide.Buy)  # 0
    ORDER_SIDE_SELL = int(vbt.portfolio.nb.OrderSide.Sell)  # 1

    # Helper to return NoOrder
    def no_order():
        return (np.nan, SIZE_TYPE_AMOUNT, ORDER_SIDE_BUY)

    # Helper to compute base units for asset A from notional
    base_units_a = 0.0
    if price_a > 0:
        base_units_a = float(notional_per_leg) / price_a
    else:
        base_units_a = 0.0

    # Establish whether we are currently in a trade (non-zero position)
    in_position = abs(pos_now) > 1e-12

    # Stop-loss: if either leg exceeds stop threshold, close positions
    if in_position and abs(z) > float(stop_threshold):
        # Close current position for this column
        size_to_close = abs(pos_now)
        if size_to_close <= 0:
            return no_order()
        # If currently long -> sell to close; if short -> buy to close
        direction = ORDER_SIDE_SELL if pos_now > 0 else ORDER_SIDE_BUY
        return (size_to_close, SIZE_TYPE_AMOUNT, direction)

    # Exit on mean reversion: z-score crosses 0 (compare previous zscore)
    if in_position and i > 0:
        prev_z = float(zscore[i - 1])
        # If previous z had different sign than current (including zero crossing), close
        # Be defensive for NaN
        if (not np.isnan(prev_z)) and (prev_z * z <= 0.0 and not (abs(prev_z) < 1e-12 and abs(z) < 1e-12)):
            size_to_close = abs(pos_now)
            if size_to_close <= 0:
                return no_order()
            direction = ORDER_SIDE_SELL if pos_now > 0 else ORDER_SIDE_BUY
            return (size_to_close, SIZE_TYPE_AMOUNT, direction)

    # Entry logic: only act if not currently in position
    if not in_position:
        # Long A, Short B when z < -entry
        if z < -float(entry_threshold):
            if col == 0:
                # Asset A: go long
                size = base_units_a
                direction = ORDER_SIDE_BUY
                return (size, SIZE_TYPE_AMOUNT, direction)
            else:
                # Asset B: go short scaled by hedge ratio (units)
                size = abs(hr) * base_units_a
                direction = ORDER_SIDE_SELL
                return (size, SIZE_TYPE_AMOUNT, direction)

        # Short A, Long B when z > entry
        if z > float(entry_threshold):
            if col == 0:
                # Asset A: go short
                size = base_units_a
                direction = ORDER_SIDE_SELL
                return (size, SIZE_TYPE_AMOUNT, direction)
            else:
                # Asset B: go long scaled by hedge ratio (units)
                size = abs(hr) * base_units_a
                direction = ORDER_SIDE_BUY
                return (size, SIZE_TYPE_AMOUNT, direction)

    # Otherwise, no order
    return no_order()
