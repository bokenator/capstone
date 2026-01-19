"""
Pairs trading strategy implementation for vectorbt backtest runner.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio with lookback (hedge_lookback)
- Spread and z-score with zscore_lookback
- Entry/exit/stop thresholds are handled in order_func
- Uses fixed notional sizing ($10,000 base per A leg) and scales B by hedge ratio
- Does not use numba
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Module-level state to coordinate multi-step ordering per bar.
# Keyed by bar index i, value is a dict with keys:
#   - 'phase': one of {'a_submitted', 'a_filled'}
#   - 'desired_a': target units for A
#   - 'desired_b': target units for B
_ORDER_STATE: Dict[int, Dict[str, Any]] = {}


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio, spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (1D numpy array).
        close_b: Prices for asset B (1D numpy array).
        hedge_lookback: Lookback window (in bars) for rolling OLS to compute hedge ratio.
        zscore_lookback: Lookback window (in bars) for rolling mean/std of the spread.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": numpy array of hedge ratios (same length as inputs)
            - "spread": numpy array of spreads (same length)
            - "zscore": numpy array of z-scores (same length)
    """
    # Basic validation and conversion
    if not isinstance(close_a, np.ndarray):
        close_a = np.asarray(close_a, dtype=float)
    if not isinstance(close_b, np.ndarray):
        close_b = np.asarray(close_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = close_a.shape[0]
    if n == 0:
        return {"hedge_ratio": np.array([]), "spread": np.array([]), "zscore": np.array([])}

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")

    # Rolling OLS regression to compute hedge ratio (slope of A ~ B)
    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        window_a = close_a[start_idx : end_idx + 1]
        window_b = close_b[start_idx : end_idx + 1]

        # If any NaNs in window, skip
        if np.isnan(window_a).any() or np.isnan(window_b).any():
            hedge_ratio[end_idx] = np.nan
            spread[end_idx] = np.nan
            continue

        try:
            lr = stats.linregress(window_b, window_a)
            slope = float(lr.slope)
        except Exception:
            slope = np.nan

        hedge_ratio[end_idx] = slope
        if not np.isnan(slope):
            spread[end_idx] = close_a[end_idx] - slope * close_b[end_idx]
        else:
            spread[end_idx] = np.nan

    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    zscore = (spread_series - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    return {"hedge_ratio": hedge_ratio, "spread": spread, "zscore": zscore}


def _cleanup_state_for_bar(i: int) -> None:
    """Helper to remove state for a bar to avoid unbounded growth."""
    try:
        _ORDER_STATE.pop(i, None)
    except Exception:
        pass


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
    """Order function for flexible multi-asset pairs strategy.

    The function coordinates orders across two assets using a simple per-bar state machine
    to avoid creating multiple order records in a single wrapper invocation (which would
    exhaust the order buffer in the runner). It emits at most one non-empty order per
    wrapper invocation and sequences A then B when opening/closing pairs.

    Sizing:
        - Base notional for A leg: $10,000
        - Units A = 10000 / price_A
        - Units B = hedge_ratio * Units A

    Returns:
        (size, size_type, direction) or (np.nan, 0, 0) for no action.
    """
    # Size type and direction integer codes (compatible with vectorbt order_nb)
    SIZE_TYPE_ABSOLUTE = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Current position for this column (units). Default to 0 if missing.
    try:
        pos_now = float(getattr(c, "position_now", 0.0))
    except Exception:
        try:
            pos_now = float(getattr(c, "position_now")[0])
        except Exception:
            pos_now = 0.0

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If essential data missing, do not trade
    if np.isnan(price_a) or np.isnan(price_b) or np.isnan(z) or np.isnan(hr):
        # Clean up old state for past bars to avoid memory leak
        _cleanup_state_for_bar(i - 1000)
        return (np.nan, 0, 0)

    # Compute previous z-score for crossing detection
    z_prev = zscore[i - 1] if i > 0 else np.nan

    # Determine desired targets (in units) for this signal
    desired_a = None
    desired_b = None

    # Stop-loss: close both
    if not np.isnan(stop_threshold) and abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    # Exit if z-score crosses the exit threshold (typically 0.0)
    elif not np.isnan(z_prev) and ((z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold)):
        desired_a = 0.0
        desired_b = 0.0
    # Entry long/short
    elif z > entry_threshold:
        base_units_a = 10000.0 / price_a
        desired_a = -base_units_a  # Short A
        desired_b = hr * base_units_a  # Long B (hedge ratio)
    elif z < -entry_threshold:
        base_units_a = 10000.0 / price_a
        desired_a = base_units_a  # Long A
        desired_b = -hr * base_units_a  # Short B
    else:
        # No trade signal; but still allow state machine to progress if present
        # If there is outstanding state for this bar, attempt to progress it
        state = _ORDER_STATE.get(i)
        if state is None:
            return (np.nan, 0, 0)
        # If state exists but no current signal, we still try to finish sequencing
        desired_a = state.get("desired_a", None)
        desired_b = state.get("desired_b", None)

    # Determine state for this bar
    state = _ORDER_STATE.get(i)

    # Helper to compute an order tuple for the current column
    def make_order_for_column(target_units: float, current_units: float) -> Tuple[float, int, int]:
        diff = target_units - current_units
        # Avoid tiny orders due to numerical noise
        if np.isclose(diff, 0.0, atol=1e-8) or np.isnan(diff):
            return (np.nan, 0, 0)
        if diff > 0:
            return (float(diff), SIZE_TYPE_ABSOLUTE, DIRECTION_LONG)
        else:
            return (float(abs(diff)), SIZE_TYPE_ABSOLUTE, DIRECTION_SHORT)

    # If there is no outstanding sequencing state, start fresh
    if state is None:
        # If signal requires simultaneous changes to both legs (which is typical),
        # we sequence: submit A first, then B in subsequent wrapper invocation(s).
        if desired_a is not None and desired_b is not None:
            # If this is the first call for this bar and column A, send A and record state
            if col == 0:
                order = make_order_for_column(desired_a, pos_now)

                # If there's an order to send for A, record that it's been submitted so that
                # the next wrapper invocation can place B once A's position is observed to have changed.
                if not np.isnan(order[0]):
                    _ORDER_STATE[i] = {
                        "phase": "a_submitted",
                        "desired_a": desired_a,
                        "desired_b": desired_b,
                    }
                    return order
                else:
                    # No A order needed; fall through to allow B in the same wrapper invocation
                    # (this can happen if A is already at target)
                    pass

            # If we reach here and either this is col==1 or A required no change, attempt to place B
            if col == 1:
                # If A was just submitted in this same wrapper invocation, we should not place B now.
                # The state would have been set above; check and suppress if so.
                if i in _ORDER_STATE and _ORDER_STATE[i].get("phase") == "a_submitted":
                    return (np.nan, 0, 0)

                # Otherwise, place B order if needed
                order = make_order_for_column(desired_b, pos_now)
                if not np.isnan(order[0]):
                    # If we place B after A (or A had no change), we can cleanup state for this bar
                    _cleanup_state_for_bar(i)
                    return order
                return (np.nan, 0, 0)

        # If only one-sided desired (shouldn't happen often), place order for the current column
        # For example, if desired_a is None but desired_b is set, then only place for B.
        if desired_a is None and desired_b is not None and col == 1:
            order = make_order_for_column(desired_b, pos_now)
            if not np.isnan(order[0]):
                _cleanup_state_for_bar(i)
                return order
            return (np.nan, 0, 0)

        if desired_b is None and desired_a is not None and col == 0:
            order = make_order_for_column(desired_a, pos_now)
            if not np.isnan(order[0]):
                _cleanup_state_for_bar(i)
                return order
            return (np.nan, 0, 0)

        # No action
        return (np.nan, 0, 0)

    # If there is an existing state, handle sequencing
    phase = state.get("phase")

    # If A was submitted previously, wait until A appears filled (observed via pos_now on col==0)
    if phase == "a_submitted":
        if col == 0:
            # We get called for col 0 first on each wrapper invocation. If A has been filled,
            # pos_now (for col 0) should be close to desired_a. If so, mark as a_filled so that
            # the subsequent col==1 call can place B.
            desired_a_state = state.get("desired_a")
            if desired_a_state is not None and np.isclose(pos_now, desired_a_state, atol=1e-8):
                # Mark filled and keep desired values for B
                _ORDER_STATE[i]["phase"] = "a_filled"
                return (np.nan, 0, 0)
            # Otherwise, no action on col 0 until filled
            return (np.nan, 0, 0)

        else:  # col == 1
            # If A was submitted and not yet observed as filled, do not place B
            return (np.nan, 0, 0)

    # If A is observed as filled, now allow B to be placed
    if phase == "a_filled":
        if col == 1:
            desired_b_state = state.get("desired_b")
            order = make_order_for_column(desired_b_state, pos_now)
            # After placing B (or if no B needed), cleanup state
            _cleanup_state_for_bar(i)
            if not np.isnan(order[0]):
                return order
            return (np.nan, 0, 0)
        else:
            # col == 0: nothing to do
            return (np.nan, 0, 0)

    # Fallback: no action
    return (np.nan, 0, 0)
