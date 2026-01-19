# Pairs trading strategy indicators and order function
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Try to get proper enum values from vectorbt if available
try:
    import vectorbt as vbt
    try:
        SizeType_Amount = int(vbt.portfolio.enums.SizeType.Amount)
        SizeType_Value = int(vbt.portfolio.enums.SizeType.Value)
    except Exception:
        SizeType_Amount = 0
        SizeType_Value = 1
    try:
        Direction_Long = int(vbt.portfolio.enums.Direction.Long)
        Direction_Short = int(vbt.portfolio.enums.Direction.Short)
    except Exception:
        Direction_Long = 1
        Direction_Short = 2
except Exception:
    SizeType_Amount = 0
    SizeType_Value = 1
    Direction_Long = 1
    Direction_Short = 2

# Globals to help coordinate multi-order placement within flexible wrapper calls
# These keep track of the wrapper inner-loop state so we don't emit duplicate orders
_SEEN_COLS: set = set()
_CREATED_ORDER_IN_INVOCATION: bool = False
_LAST_INVOCATION_INDEX: int = -999999
_NUM_COLS: int = 2
_IN_TRADE: bool = False
_LAST_ENTRY_INDEX: int = -999999
_MIN_HOLD_BARS: int = 1

# Clip hedge ratio to avoid extreme position sizing due to noisy regressions
_HEDGE_RATIO_CLIP = 5.0


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    This function is flexible in the input types:
      - If either input is a pandas Series/DataFrame, outputs are returned as pandas Series
        aligned to the (intersected) index of the inputs.
      - If both inputs are array-like (numpy), outputs are numpy arrays.

    Args:
        close_a: Prices for asset A (1D array-like or pandas Series/DataFrame).
        close_b: Prices for asset B (1D array-like or pandas Series/DataFrame).
        hedge_lookback: Maximum window length for rolling OLS (uses fewer points until available).
        zscore_lookback: Window length for rolling mean/std used in z-score (uses fewer points until available).

    Returns:
        Dict with keys:
            - 'hedge_ratio': array-like of hedge ratios (slope a ~ b)
            - 'spread': array-like of spread values price_a - hedge_ratio * price_b
            - 'zscore': array-like of z-scores

    Notes:
        - No lookahead: all calculations for index t only use data up to and including t.
        - For early indices where full lookbacks are not available, expanding windows are used.
        - Avoids NaNs by falling back to previous hedge ratio and small epsilon for zero std.
    """

    # Helper to coerce inputs to pandas Series when appropriate
    def _to_series(x: Any) -> pd.Series:
        if isinstance(x, pd.Series):
            return x.copy()
        if isinstance(x, pd.DataFrame):
            # If DataFrame has a single column, return it. Otherwise, take the first column.
            if x.shape[1] == 1:
                return x.iloc[:, 0].copy()
            return x.iloc[:, 0].copy()
        # For numpy/other array-like, create a RangeIndex to be consistent
        return pd.Series(np.asarray(x).ravel())

    is_pandas_input = isinstance(close_a, (pd.Series, pd.DataFrame)) or isinstance(close_b, (pd.Series, pd.DataFrame))

    if is_pandas_input:
        sa = _to_series(close_a)
        sb = _to_series(close_b)

        # Align on index (inner join) to avoid shape mismatch when truncated data provided
        df = pd.concat([sa, sb], axis=1, join="inner")
        if df.shape[1] < 2:
            # If concatenation failed to provide two columns, fallback to aligning by position
            a_vals = np.asarray(sa).ravel()
            b_vals = np.asarray(sb).ravel()
            L = min(a_vals.size, b_vals.size)
            a_vals = a_vals[:L]
            b_vals = b_vals[:L]
            index = None
        else:
            a_vals = df.iloc[:, 0].to_numpy(dtype=float)
            b_vals = df.iloc[:, 1].to_numpy(dtype=float)
            index = df.index

    else:
        # Numpy path: coerce arrays and align by shortest length if mismatch
        a_vals = np.asarray(close_a, dtype=float).ravel()
        b_vals = np.asarray(close_b, dtype=float).ravel()
        if a_vals.size != b_vals.size:
            L = min(a_vals.size, b_vals.size)
            a_vals = a_vals[:L]
            b_vals = b_vals[:L]
        index = None

    if a_vals.shape != b_vals.shape:
        raise ValueError("close_a and close_b must have the same shape after alignment")

    n = a_vals.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Ensure lookbacks are sensible
    hedge_lookback = max(1, int(hedge_lookback))
    zscore_lookback = max(1, int(zscore_lookback))

    # Rolling OLS (a ~ b) using only past and current data (no future data)
    last_valid_hr = 1.0  # fallback hedge ratio
    for t in range(n):
        start = max(0, t - hedge_lookback + 1)
        y = a_vals[start : t + 1]
        x = b_vals[start : t + 1]

        # Need at least 2 points to fit slope; otherwise reuse last
        if y.size >= 2 and np.isfinite(y).all() and np.isfinite(x).all():
            try:
                res = linregress(x, y)
                slope = float(res.slope)
                if not np.isfinite(slope):
                    slope = last_valid_hr
                else:
                    last_valid_hr = slope
            except Exception:
                slope = last_valid_hr
        else:
            slope = last_valid_hr

        # Clip extreme hedge ratios to avoid very large position sizes from noisy early regressions
        slope = float(np.clip(slope, -_HEDGE_RATIO_CLIP, _HEDGE_RATIO_CLIP))
        hedge_ratio[t] = slope

        # Compute spread at time t
        if np.isfinite(a_vals[t]) and np.isfinite(b_vals[t]) and np.isfinite(slope):
            spread[t] = a_vals[t] - slope * b_vals[t]
        else:
            spread[t] = np.nan

    # Rolling mean and std for z-score (expanding until window full)
    eps = 1e-8
    for t in range(n):
        start = max(0, t - zscore_lookback + 1)
        window = spread[start : t + 1]
        # consider only finite values
        finite_win = window[np.isfinite(window)]
        if finite_win.size == 0:
            zscore[t] = np.nan
            continue
        mu = finite_win.mean()
        sigma = finite_win.std(ddof=0)
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = eps
        if np.isfinite(spread[t]):
            zscore[t] = (spread[t] - mu) / sigma
        else:
            zscore[t] = np.nan

    # Return as pandas Series if inputs were pandas; otherwise numpy arrays
    if is_pandas_input and index is not None:
        return {
            "hedge_ratio": pd.Series(hedge_ratio, index=index),
            "spread": pd.Series(spread, index=index),
            "zscore": pd.Series(zscore, index=index),
        }

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
) -> Tuple[float, int, int]:
    """
    Order function for a 2-asset pairs trading strategy (flexible mode).

    Returns a tuple (size, size_type, direction) where:
        - size: positive float
        - size_type: int (0 => amount/units)
        - direction: int (1 = long/buy, 2 = short/sell)

    Logic:
        - Entry when zscore crosses the entry threshold (prevents repeated re-entries while already above threshold)
            - z > +entry: Short A, Long B (A: sell, B: buy)
            - z < -entry: Long A, Short B (A: buy, B: sell)
        - Exit when zscore crosses zero or when |zscore| > stop_threshold (stop loss)
        - Position sizing: base_amount = 10_000 / price_A; size_B = |hedge_ratio| * base_amount (units)
    """
    global _SEEN_COLS, _CREATED_ORDER_IN_INVOCATION, _LAST_INVOCATION_INDEX, _IN_TRADE, _LAST_ENTRY_INDEX

    # Safely get index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Reset seen state at the start of a new wrapper invocation (same bar but new call)
    if _LAST_INVOCATION_INDEX != i or len(_SEEN_COLS) == _NUM_COLS:
        _SEEN_COLS.clear()
        _CREATED_ORDER_IN_INVOCATION = False
        _LAST_INVOCATION_INDEX = i

    # Current position for this asset (units)
    position_now = float(getattr(c, "position_now", 0.0))

    # Basic validations
    # Accept zscore and hedge_ratio as pandas Series or numpy arrays
    if isinstance(zscore, (pd.Series, pd.DataFrame)):
        z_arr = zscore.values.ravel()
    else:
        z_arr = np.asarray(zscore).ravel()
    if isinstance(hedge_ratio, (pd.Series, pd.DataFrame)):
        hr_arr = hedge_ratio.values.ravel()
    else:
        hr_arr = np.asarray(hedge_ratio).ravel()

    n = len(z_arr)
    if i < 0 or i >= n:
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = float(z_arr[i])
    hr = float(hr_arr[i])

    # Clip hedge ratio to avoid extreme sizing
    if not np.isfinite(hr):
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)
    hr = float(np.clip(hr, -_HEDGE_RATIO_CLIP, _HEDGE_RATIO_CLIP))

    # If indicators or prices invalid, do nothing
    if not np.isfinite(z) or price_a <= 0 or price_b <= 0:
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)

    # sizing (amount-based, rounded to integer units to avoid fractional partial fills)
    notional = 10_000.0
    units_a = max(1, int(round(notional / price_a)))
    units_b = max(1, int(round(abs(hr) * units_a)))

    # thresholds
    stop = abs(z) > stop_threshold

    # detect zero crossing (use previous zscore only)
    crossed_zero = False
    prev_z = None
    if i > 0 and np.isfinite(z_arr[i - 1]):
        prev_z = float(z_arr[i - 1])
        if (prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0):
            crossed_zero = True

    # detect entry crossing to avoid repeated entries
    entry_cross = False
    if abs(z) > entry_threshold:
        if i == 0:
            entry_cross = True
        else:
            prev = z_arr[i - 1]
            if not np.isfinite(prev) or abs(prev) <= entry_threshold:
                entry_cross = True

    # Priority: stop-loss (close if in position), then exit-on-reversion, then entry
    # We emit at most one order per wrapper invocation; if we've already emitted one, skip further orders
    if _CREATED_ORDER_IN_INVOCATION:
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)

    # Close logic: if this asset has a position, close it when stop or crossed_zero
    if (stop or crossed_zero) and abs(position_now) > 0:
        # enforce min holding constraint for non-stop exits
        if not stop and (_LAST_ENTRY_INDEX is not None) and (i - _LAST_ENTRY_INDEX) < _MIN_HOLD_BARS:
            # skip closing until minimum holding reached
            _SEEN_COLS.add(col)
            return (np.nan, 0, 0)

        size_to_close = abs(position_now)
        direction = Direction_Short if position_now > 0 else Direction_Long
        _CREATED_ORDER_IN_INVOCATION = True
        _IN_TRADE = False
        _SEEN_COLS.add(col)
        return (float(size_to_close), SizeType_Amount, int(direction))

    # Do not enter if already in position for this asset
    if abs(position_now) > 0:
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)

    # Do not enter if we're beyond stop threshold
    if stop:
        _SEEN_COLS.add(col)
        return (np.nan, 0, 0)

    # Entry logic only when crossing threshold. Only emit one order per invocation to let engine process
    # the first leg, then the second leg on the next wrapper invocation.
    if entry_cross and not _IN_TRADE:
        if col == 0:
            # Emit order for asset A first
            if z > entry_threshold:
                # Short A (sell by units)
                _CREATED_ORDER_IN_INVOCATION = True
                _IN_TRADE = True
                _LAST_ENTRY_INDEX = i
                _SEEN_COLS.add(col)
                return (float(units_a), SizeType_Amount, int(Direction_Short))
            elif z < -entry_threshold:
                # Long A (buy by units)
                _CREATED_ORDER_IN_INVOCATION = True
                _IN_TRADE = True
                _LAST_ENTRY_INDEX = i
                _SEEN_COLS.add(col)
                return (float(units_a), SizeType_Amount, int(Direction_Long))
        else:
            # col == 1: second leg placed in a later invocation
            if z > entry_threshold:
                # Long B (buy by units scaled by hedge ratio)
                _CREATED_ORDER_IN_INVOCATION = True
                _SEEN_COLS.add(col)
                return (float(units_b), SizeType_Amount, int(Direction_Long))
            elif z < -entry_threshold:
                # Short B (sell by units)
                _CREATED_ORDER_IN_INVOCATION = True
                _SEEN_COLS.add(col)
                return (float(units_b), SizeType_Amount, int(Direction_Short))

    # No order
    _SEEN_COLS.add(col)
    return (np.nan, 0, 0)
