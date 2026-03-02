"""
MACD + ATR-based trailing stop strategy implementation.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
    -> returns dict[str, np.ndarray] with keys: 'close','high','macd','signal','atr','sma'

- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)
    -> called by vectorbt.from_order_func (use_numba=False). Returns tuple (size, size_type, direction)

Notes:
- No numba usage.
- Uses simple integer codes for size_type/direction (to be interpreted by the provided wrapper):
    SIZE_TYPE_TARGET_PERCENT = 3  # target percent allocation (set 1.0 to allocate 100%)
    DIRECTION_LONG = 1
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_BOTH = 0

The order_func keeps a small per-column state to track highest price since entry for trailing stop.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Local order state (per-column) to track entry/highest price since entry when vectorbt's context
# doesn't expose entry index. Keyed by column index (int).
_ORDER_STATE: Dict[int, Dict[str, Any]] = {}

# Constants for size_type and direction. We avoid importing vbt enums here per instructions.
SIZE_TYPE_TARGET_PERCENT = 3  # assumed mapping for TargetPercent
DIRECTION_LONG = 1
SIZE_TYPE_AMOUNT = 0
DIRECTION_BOTH = 0


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute MACD, signal line, SMA and ATR.

    Returns a dictionary with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    Args:
        ohlcv: DataFrame with at least 'high','low','close' columns.
        macd_fast, macd_slow, macd_signal: MACD EMAs.
        sma_period: Period for trend filter SMA.
        atr_period: Period for ATR.
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    # MACD
    # Use pandas' EWM (adjust=False) for standard EMA calculation
    if macd_fast <= 0 or macd_slow <= 0 or macd_signal <= 0:
        raise ValueError("MACD periods must be positive integers")

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR (True Range then Wilder's smoothing via EWM with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Use Wilder's smoothing (alpha=1/atr_period)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    # Convert to numpy arrays (keep np.nan where data is insufficient)
    out = {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
    }

    return out


def _get_c_attr(c: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    """Helper to retrieve first available attribute from context 'c'."""
    for nm in names:
        if hasattr(c, nm):
            return getattr(c, nm)
    return default


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """Order function for vectorbt.from_order_func (use_numba=False).

    Logic (long-only):
    - Entry when MACD crosses above Signal AND price > 50-period SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction). Returning size=np.nan indicates no action.

    Note: We use a small module-level state to track highest price since entry per column.
    """
    # Get current index and column (if provided). Default column = 0 for single-asset.
    i = int(_get_c_attr(c, ("i", "idx", "index"), 0))
    col = int(_get_c_attr(c, ("col", "column", "column_index"), 0))

    # Ensure arrays are numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    n = len(close)
    if i < 0 or i >= n:
        # Invalid index -> no action
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Prepare per-column state
    state = _ORDER_STATE.setdefault(col, {"in_pos": False, "highest": -np.inf, "entry_idx": None})

    # Try to infer whether we are currently in a position from the context if possible.
    # Prefer numeric 'pos' attribute; fall back to boolean 'is_open'/'is_long' if available.
    pos_val = _get_c_attr(c, ("pos", "position", "size", "position_size"), None)
    is_open_val = _get_c_attr(c, ("is_open", "is_long", "has_long", "open"), None)

    in_pos = None
    if pos_val is not None:
        try:
            in_pos = bool(float(pos_val) != 0.0)
        except Exception:
            in_pos = bool(pos_val)
    elif is_open_val is not None:
        in_pos = bool(is_open_val)
    else:
        # Fall back to stored state
        in_pos = bool(state.get("in_pos", False))

    # Detect MACD cross events safely (require previous bar to exist and be finite)
    def cross_up(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx], arr_b[idx]
        a1, b1 = arr_a[idx - 1], arr_b[idx - 1]
        if not (np.isfinite(a0) and np.isfinite(b0) and np.isfinite(a1) and np.isfinite(b1)):
            return False
        return (a0 > b0) and (a1 <= b1)

    def cross_down(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx], arr_b[idx]
        a1, b1 = arr_a[idx - 1], arr_b[idx - 1]
        if not (np.isfinite(a0) and np.isfinite(b0) and np.isfinite(a1) and np.isfinite(b1)):
            return False
        return (a0 < b0) and (a1 >= b1)

    # Evaluate indicator conditions
    price = float(close[i]) if np.isfinite(close[i]) else np.nan

    sma_ok = np.isfinite(sma[i]) and price > float(sma[i])
    macd_cross_up = cross_up(macd, signal, i)
    macd_cross_down = cross_down(macd, signal, i)

    # Update highest since entry if we are currently in position
    # Prefer getting entry index from context if available
    entry_idx_ctx = _get_c_attr(c, ("entry_idx", "entry_index", "entry_i", "entry_row"), None)

    if in_pos:
        # Determine entry index
        entry_idx = None
        if entry_idx_ctx is not None and entry_idx_ctx is not False:
            try:
                entry_idx = int(entry_idx_ctx)
                if entry_idx < 0:
                    entry_idx = None
            except Exception:
                entry_idx = None

        if entry_idx is not None:
            # compute highest from recorded entry index, but don't slice out-of-bounds
            start = max(0, entry_idx)
            if start <= i:
                highest_since_entry = float(np.nanmax(high[start : i + 1]))
            else:
                highest_since_entry = float(high[i])
            state["highest"] = highest_since_entry
            state["entry_idx"] = entry_idx
            state["in_pos"] = True
        else:
            # Use/update stored highest
            prev_highest = state.get("highest", -np.inf)
            if not np.isfinite(prev_highest) or prev_highest == -np.inf:
                # Initialize if missing
                prev_highest = float(high[i])
            # Update with current bar high
            state["highest"] = max(float(prev_highest), float(high[i]))
            state["in_pos"] = True
    else:
        # Not in position: ensure state reset
        state.setdefault("highest", -np.inf)
        # Do not clear entry_idx immediately because c may report entry_idx on bar of entry; keep behavior conservative
        state["in_pos"] = False

    # Construct trailing stop price if possible
    ts_price = None
    if state.get("highest", None) is not None and np.isfinite(state.get("highest", np.nan)) and np.isfinite(atr[i]):
        ts_price = state["highest"] - float(trailing_mult) * float(atr[i])

    # Exit condition: either MACD cross down OR price falls below trailing stop
    exit_by_trailing = False
    if in_pos and (ts_price is not None) and np.isfinite(price):
        exit_by_trailing = price < ts_price

    # Decide orders
    # Entry: only if currently flat (not in position), and macd cross up and price > SMA
    if (not in_pos) and macd_cross_up and sma_ok:
        # Place order to target 100% (long-only)
        # Update state proactively
        state["in_pos"] = True
        state["entry_idx"] = i
        state["highest"] = float(high[i])
        return (1.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_LONG)

    # Exit: if in position and either macd cross down or trailing stop triggered
    if in_pos and (macd_cross_down or exit_by_trailing):
        # Close position by targeting 0% allocation
        state["in_pos"] = False
        # Keep last highest for reference; reset on next flat
        return (0.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_LONG)

    # No action
    return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
