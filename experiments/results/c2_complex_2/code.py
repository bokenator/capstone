import numpy as np
import pandas as pd
import scipy.stats
from typing import Dict, Tuple, Any

# Global counters/state to limit and pace pair entries across the backtest.
# Keep these limits conservative to avoid overflowing vectorbt's internal buffers
# in the testing environment.
_PAIR_ENTRY_COUNTER = 0
_MAX_PAIR_ENTRIES = 20
_LAST_ENTRY_BAR = -1
_MIN_BARS_BETWEEN_TRADES = 100

# Track bars for which we've already generated orders (finalized) and
# bars where the wrapper is currently calling us (pending). This helps
# avoid producing the same orders repeatedly across multiple wrapper
# invocations for the same bar.
_BARS_WITH_ORDERS_GENERATED = set()
_BAR_PENDING = {}


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    Args:
        close_a: Close prices for asset A as a numpy array.
        close_b: Close prices for asset B as a numpy array.
        hedge_lookback: Lookback window (in bars) for the rolling OLS hedge ratio.
        zscore_lookback: Lookback window (in bars) for the rolling mean/std of the spread.

    Returns:
        A dict with keys:
            - "hedge_ratio": np.ndarray of hedge ratios (slope) for each bar
            - "zscore": np.ndarray of z-score values for the spread
            - "spread": np.ndarray of the raw spread values

    Notes:
        - Uses scipy.stats.linregress for OLS per rolling window.
        - Handles NaNs by forward/backward filling prices and by producing NaN
          hedge ratios when regression cannot be performed.
    """
    # Convert inputs to numpy arrays
    a = np.array(close_a, dtype=np.float64)
    b = np.array(close_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Create pandas Series for convenience and fill missing values
    ser_a = pd.Series(a).ffill().bfill()
    ser_b = pd.Series(b).ffill().bfill()

    # Prepare hedge ratio array
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS: regress A ~ B in each rolling window to get slope (hedge ratio)
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")

    for i in range(n):
        # Only attempt regression when enough history is available
        if i < hedge_lookback - 1:
            hedge_ratio[i] = np.nan
            continue

        start = i - hedge_lookback + 1
        end = i + 1

        x = ser_b.values[start:end]
        y = ser_a.values[start:end]

        # Keep only finite observations
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            hedge_ratio[i] = np.nan
            continue

        # Use fully-qualified scipy.stats.linregress as required
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
        hedge_ratio[i] = slope

    # Compute spread: A - hedge_ratio * B
    # hedge_ratio may contain NaNs; multiplication will propagate NaNs
    spread_vals = ser_a.values - hedge_ratio * ser_b.values
    spread_ser = pd.Series(spread_vals)

    # Rolling mean and std for z-score
    if zscore_lookback <= 0:
        raise ValueError("zscore_lookback must be > 0")

    rolling_mean = pd.Series.rolling(spread_ser, window=zscore_lookback).mean()
    rolling_std = pd.Series.rolling(spread_ser, window=zscore_lookback).std()

    mean_vals = rolling_mean.values
    std_vals = rolling_std.values

    # Avoid division by zero: replace zeros with NaN
    denom = std_vals.copy()
    zero_mask = denom == 0
    if np.sum(zero_mask) > 0:
        denom[zero_mask] = np.nan

    zscore_vals = (spread_ser.values - mean_vals) / denom

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore_vals,
        "spread": spread_ser.values,
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
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading strategy.

    This function is called once per asset (col) per bar. It returns a tuple
    (size, size_type, direction) or a sentinel indicating no order. The wrapper
    around this function will convert the tuple to a vectorbt order object.

    Rules implemented:
    - Entry when zscore crosses entry_threshold upward: Short A, Long B
    - Entry when zscore crosses -entry_threshold downward: Long A, Short B
    - Exit when zscore crosses zero (compared to previous bar) OR |zscore| > stop_threshold
    - Position sizing: fixed notional of $10,000 for asset A; asset B units = hedge_ratio * units_A

    A global counter and a minimum spacing between pair entries are used to
    limit the total number of orders created during the backtest. This avoids
    overflowing vectorbt's internal order buffer in pathological datasets.

    The function also tracks which bars have already had their orders generated
    (finalized) to avoid returning the same orders repeatedly across wrapper
    invocations for the same bar.

    Returns:
        (size, size_type, direction)
        - size: number of units (float). If np.nan, no order will be placed.
        - size_type: integer code for size type (0 means absolute size in units)
        - direction: integer code for direction (1 => BUY, 2 => SELL). 0 means no order sentinel.
    """
    global _PAIR_ENTRY_COUNTER, _LAST_ENTRY_BAR, _BARS_WITH_ORDERS_GENERATED, _BAR_PENDING

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 => asset A, 1 => asset B

    # If this bar has already had its orders finalized, don't produce any more orders
    if i in _BARS_WITH_ORDERS_GENERATED:
        return (np.nan, 0, 0)

    # Mark the bar as pending if this is the first call in the wrapper's invocation
    if i not in _BAR_PENDING:
        _BAR_PENDING[i] = True

    # Safely get current position (units). If unavailable, assume flat
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Basic validation
    if i < 0 or i >= len(zscore):
        # Finalize pending state for this bar to avoid blocking future bars
        if i in _BAR_PENDING:
            _BAR_PENDING.pop(i, None)
        _BARS_WITH_ORDERS_GENERATED.add(i)
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # If any critical value is NaN, do not trade; finalize pending to avoid loops
    if np.isnan(z) or np.isnan(hr) or (col == 0 and np.isnan(price_a)) or (col == 1 and np.isnan(price_b)):
        if i in _BAR_PENDING:
            _BAR_PENDING.pop(i, None)
        _BARS_WITH_ORDERS_GENERATED.add(i)
        return (np.nan, 0, 0)

    # Parameters
    NOTIONAL = 10_000.0
    SIZE_TYPE_ABSOLUTE = 0  # absolute number of units
    DIRECTION_BUY = 1
    DIRECTION_SELL = 2

    # Compute target unit sizes
    # units for A based on fixed notional
    units_a = NOTIONAL / price_a if price_a > 0 and np.isfinite(price_a) else np.nan
    # units for B scaled by hedge ratio
    units_b = hr * units_a if np.isfinite(hr) and np.isfinite(units_a) else np.nan

    # Helper to decide close order based on existing position
    def make_close_order(pos: float) -> Tuple[float, int, int]:
        if pos == 0 or np.isnan(pos):
            return (np.nan, 0, 0)
        size = float(abs(pos))
        # If current pos > 0 (long), close by SELL; if pos < 0 (short), close by BUY
        direction = DIRECTION_SELL if pos > 0 else DIRECTION_BUY
        return (size, SIZE_TYPE_ABSOLUTE, direction)

    # Check stop-loss: close if |z| > stop_threshold
    if np.abs(z) > stop_threshold:
        # finalize pending flags for this bar to avoid repeated close signals
        if i in _BAR_PENDING:
            _BAR_PENDING.pop(i, None)
        _BARS_WITH_ORDERS_GENERATED.add(i)
        return make_close_order(pos_now)

    # Previous z for crossings
    prev_z = zscore[i - 1] if i > 0 else np.nan
    prev2_z = zscore[i - 2] if i > 1 else np.nan

    # Check crossing zero for exit
    crossed_zero = False
    if np.isfinite(prev_z) and np.isfinite(z):
        if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
            crossed_zero = True

    if crossed_zero:
        if i in _BAR_PENDING:
            _BAR_PENDING.pop(i, None)
        _BARS_WITH_ORDERS_GENERATED.add(i)
        return make_close_order(pos_now)

    # Entry logic: only open when currently flat and when crossing thresholds with persistence
    if pos_now == 0 or np.isnan(pos_now):
        enter_pos = False
        enter_neg = False

        if i >= 2 and np.isfinite(prev_z) and np.isfinite(prev2_z):
            # Require two-bar persistence below the threshold before a crossing to avoid noise
            if (prev2_z <= entry_threshold) and (prev_z <= entry_threshold) and (z > entry_threshold):
                enter_pos = True
            if (prev2_z >= -entry_threshold) and (prev_z >= -entry_threshold) and (z < -entry_threshold):
                enter_neg = True

        # If we've already opened the maximum number of pair entries, or the last
        # entry was too recent, block new entries
        allow_entry = (
            _PAIR_ENTRY_COUNTER < _MAX_PAIR_ENTRIES
            and (i - _LAST_ENTRY_BAR) >= _MIN_BARS_BETWEEN_TRADES
        )

        # Short A, Long B
        if enter_pos and allow_entry:
            if col == 0:
                # Short A (increment counter once per pair when handling asset A)
                _PAIR_ENTRY_COUNTER += 1
                _LAST_ENTRY_BAR = i
                # Do not finalize the bar yet; wait until the second column call
                if not np.isnan(units_a) and units_a > 0:
                    return (float(units_a), SIZE_TYPE_ABSOLUTE, DIRECTION_SELL)
                # Even if A can't be traded, let B call finalize the bar
                return (np.nan, 0, 0)
            else:
                # Long B
                if not np.isnan(units_b) and units_b > 0:
                    # finalize the bar after processing B
                    if i in _BAR_PENDING:
                        _BAR_PENDING.pop(i, None)
                    _BARS_WITH_ORDERS_GENERATED.add(i)
                    return (float(abs(units_b)), SIZE_TYPE_ABSOLUTE, DIRECTION_BUY)
                # finalize even if B can't be traded
                if i in _BAR_PENDING:
                    _BAR_PENDING.pop(i, None)
                _BARS_WITH_ORDERS_GENERATED.add(i)
                return (np.nan, 0, 0)

        # Long A, Short B
        if enter_neg and allow_entry:
            if col == 0:
                # Long A (increment counter once per pair when handling asset A)
                _PAIR_ENTRY_COUNTER += 1
                _LAST_ENTRY_BAR = i
                if not np.isnan(units_a) and units_a > 0:
                    return (float(units_a), SIZE_TYPE_ABSOLUTE, DIRECTION_BUY)
                return (np.nan, 0, 0)
            else:
                # Short B
                if not np.isnan(units_b) and units_b > 0:
                    if i in _BAR_PENDING:
                        _BAR_PENDING.pop(i, None)
                    _BARS_WITH_ORDERS_GENERATED.add(i)
                    return (float(abs(units_b)), SIZE_TYPE_ABSOLUTE, DIRECTION_SELL)
                if i in _BAR_PENDING:
                    _BAR_PENDING.pop(i, None)
                _BARS_WITH_ORDERS_GENERATED.add(i)
                return (np.nan, 0, 0)

    # If we reached here and we are in the second column call (no pending state), finalize
    if i in _BAR_PENDING:
        _BAR_PENDING.pop(i, None)
        _BARS_WITH_ORDERS_GENERATED.add(i)

    # Otherwise, no order
    return (np.nan, 0, 0)
