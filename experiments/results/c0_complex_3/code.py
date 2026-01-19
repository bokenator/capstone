# pairs_strategy.py
"""
Pairs trading strategy implementation for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Hedge ratio computed with rolling OLS (scipy.stats.linregress) over hedge_lookback
- Z-score computed with rolling mean/std over zscore_lookback
- Position sizing: fixed notional $10,000 per leg
- No use of numba
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

# State to coordinate multi-leg order creation across wrapper invocations per bar
_BAR_ORDER_STATE: Dict[int, Dict[str, Any]] = {}
# Global counter to prevent runaway order creation and hitting internal limits
_ORDER_COUNTER: int = 0
# Keep global cap safely below typical vectorbt internal max_orders
_MAX_GLOBAL_ORDERS: int = 900


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio, spread and z-score for a pairs trading strategy.

    Args:
        close_a: 1D array of close prices for asset A
        close_b: 1D array of close prices for asset B
        hedge_lookback: lookback for rolling OLS to compute hedge ratio
        zscore_lookback: lookback for rolling mean/std of the spread to compute z-score

    Returns:
        Dict with keys: 'hedge_ratio', 'spread', 'zscore' mapping to numpy arrays
    """
    # Convert inputs to numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if len(close_a) != len(close_b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(close_a)

    # Initialize hedge ratio with NaNs
    hedge_ratio = np.full(n, np.nan)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling OLS to estimate hedge ratio (slope of regression of A on B)
    # For each window ending at index i, compute slope if window has no NaNs and non-zero variance
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        window_a = close_a[start_idx : end_idx + 1]
        window_b = close_b[start_idx : end_idx + 1]

        # Skip windows with NaNs
        if np.isnan(window_a).any() or np.isnan(window_b).any():
            continue

        # Skip if constant series in B (cannot regress)
        if np.allclose(window_b, window_b[0]):
            continue

        # Regress A ~ B (slope is hedge ratio)
        try:
            res = linregress(window_b, window_a)
            hedge_ratio[end_idx] = float(res.slope)
        except Exception:
            # In case regression fails for any reason, leave NaN
            hedge_ratio[end_idx] = np.nan

    # Compute spread: A - beta * B
    # This will yield NaN where hedge_ratio is NaN
    spread = close_a - hedge_ratio * close_b

    # Compute rolling mean and std of spread for z-score
    spread_series = pd.Series(spread)

    # Use min_periods equal to full lookback to respect warmup
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    roll_mean_arr = roll_mean.to_numpy(dtype=float)
    roll_std_arr = roll_std.to_numpy(dtype=float)

    # Compute z-score with protection for zero std
    zscore = np.full(n, np.nan)
    valid = (~np.isnan(spread)) & (~np.isnan(roll_mean_arr)) & (~np.isnan(roll_std_arr)) & (roll_std_arr > 0)
    zscore[valid] = (spread[valid] - roll_mean_arr[valid]) / roll_std_arr[valid]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
    }


def _cleanup_bar_state(current_i: int) -> None:
    """Remove very old entries from bar order state to avoid memory growth."""
    # Keep only recent 10 bars
    keys_to_remove = [k for k in _BAR_ORDER_STATE.keys() if k < current_i - 10]
    for k in keys_to_remove:
        _BAR_ORDER_STATE.pop(k, None)


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, int, int]:
    """
    Order function for a pairs trading strategy (flexible multi-asset compatible).

    This function returns a tuple (size, size_type, direction).
    - size: positive float number of units to trade (np.nan for no order)
    - size_type: integer code for size type (0 = absolute number of shares)
    - direction: integer (1 = buy/long, 2 = sell/short)

    The logic follows:
    - Entry when zscore crosses entry_threshold from the opposite side:
        * z_prev < entry_threshold and z >= entry_threshold -> Short A, Long B
        * z_prev > -entry_threshold and z <= -entry_threshold -> Long A, Short B
    - Exit when zscore crosses zero or when |zscore| > stop_threshold (stop-loss)

    Position sizing:
    - Fixed notional per leg: $10,000
    - A size = notional / price_A
    - B size = |hedge_ratio| * (notional / price_B)  (scaled by hedge ratio to approximate hedge)

    Returning state is coordinated across wrapper invocations to create at most one
    order record per wrapper call (prevents creating multiple order_records in the same call).
    A global cap limits the total number of orders to avoid filling internal buffers.
    """
    global _BAR_ORDER_STATE, _ORDER_COUNTER

    # Ensure arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    # Basic context values
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))
    cash_now = float(getattr(c, "cash_now", np.nan))

    # Cleanup old state
    _cleanup_bar_state(i)

    # Initialize per-bar state
    state = _BAR_ORDER_STATE.get(i)
    if state is None:
        state = {"cols": set(), "created_cash": None}
        _BAR_ORDER_STATE[i] = state

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # Constants
    NOTIONAL = 10_000.0
    SIZE_TYPE_ABS = 0  # use absolute number of shares
    DIR_BUY = 1
    DIR_SELL = 2
    EPS_POS = 1e-12

    # If we don't have required indicator data, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price) or price <= 0:
        return (np.nan, 0, 0)

    # Previous value for entry/exit detection
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Entry signals defined as threshold crossings (to avoid repeated entries every bar)
    open_short_a = False
    open_long_a = False
    if not np.isnan(z_prev):
        if (z_prev < entry_threshold) and (z >= entry_threshold):
            open_short_a = True
        if (z_prev > -entry_threshold) and (z <= -entry_threshold):
            open_long_a = True

    # STOP-LOSS: if |z| > stop_threshold -> close existing positions (or do nothing if none)
    if abs(z) > stop_threshold:
        if abs(pos_now) > EPS_POS and _ORDER_COUNTER < _MAX_GLOBAL_ORDERS:
            # Close current position by trading the absolute position size in opposite direction
            size = abs(pos_now)
            direction = DIR_SELL if pos_now > 0 else DIR_BUY  # sell to close long, buy to close short
            # Record that we created an order for this bar/col
            state["cols"].add(col)
            state["created_cash"] = state["created_cash"] if state["created_cash"] is not None else cash_now
            _ORDER_COUNTER += 1
            return (float(size), int(SIZE_TYPE_ABS), int(direction))
        # If not in position or limit reached, do not open/close
        return (np.nan, 0, 0)

    # EXIT: z-score crosses zero -> close both legs
    crossed_zero = False
    if not np.isnan(z_prev):
        if z == 0.0 or z_prev == 0.0 or (z_prev * z < 0):
            crossed_zero = True
    if crossed_zero and abs(pos_now) > EPS_POS and _ORDER_COUNTER < _MAX_GLOBAL_ORDERS:
        size = abs(pos_now)
        direction = DIR_SELL if pos_now > 0 else DIR_BUY
        state["cols"].add(col)
        state["created_cash"] = state["created_cash"] if state["created_cash"] is not None else cash_now
        _ORDER_COUNTER += 1
        return (float(size), int(SIZE_TYPE_ABS), int(direction))

    # Do not open new trades if we are already in a position for this asset
    if abs(pos_now) > EPS_POS:
        return (np.nan, 0, 0)

    # ENTRY: only on threshold crossing (to avoid repeated entries each bar)
    # We coordinate multi-leg creation across wrapper invocations using _BAR_ORDER_STATE
    if (open_short_a or open_long_a):
        # Enforce global cap
        if _ORDER_COUNTER >= _MAX_GLOBAL_ORDERS:
            return (np.nan, 0, 0)

        # If this exact col already has an order created, skip
        if col in state["cols"]:
            return (np.nan, 0, 0)

        # If no orders have been created yet for this bar -> allow current col to create an order
        if len(state["cols"]) == 0:
            # Create and record
            state["cols"].add(col)
            state["created_cash"] = cash_now
            _ORDER_COUNTER += 1
            # Create order for appropriate side
            if open_short_a:
                if col == 0:
                    size = NOTIONAL / price
                    direction = DIR_SELL
                else:
                    size = abs(hr) * (NOTIONAL / price)
                    direction = DIR_BUY
            else:  # open_long_a
                if col == 0:
                    size = NOTIONAL / price
                    direction = DIR_BUY
                else:
                    size = abs(hr) * (NOTIONAL / price)
                    direction = DIR_SELL

            return (float(size), int(SIZE_TYPE_ABS), int(direction))

        # If one order was already created earlier for this bar, only allow a second order if
        # the first order has been executed (cash changed since creation). This prevents creating
        # both order records in the same wrapper invocation.
        if len(state["cols"]) == 1:
            created_cash = state.get("created_cash")
            # If created_cash is None (shouldn't happen), be conservative and skip
            if created_cash is None:
                return (np.nan, 0, 0)
            # If cash_now differs from created_cash, the previous order was executed -> allow second
            if not np.isclose(cash_now, created_cash) and _ORDER_COUNTER < _MAX_GLOBAL_ORDERS:
                # allow creating for the other leg (if not already created)
                state["cols"].add(col)
                state["created_cash"] = cash_now
                _ORDER_COUNTER += 1
                if open_short_a:
                    if col == 0:
                        size = NOTIONAL / price
                        direction = DIR_SELL
                    else:
                        size = abs(hr) * (NOTIONAL / price)
                        direction = DIR_BUY
                else:  # open_long_a
                    if col == 0:
                        size = NOTIONAL / price
                        direction = DIR_BUY
                    else:
                        size = abs(hr) * (NOTIONAL / price)
                        direction = DIR_SELL

                return (float(size), int(SIZE_TYPE_ABS), int(direction))

        # Otherwise, skip creating an order now
        return (np.nan, 0, 0)

    # Default: no order
    return (np.nan, 0, 0)
