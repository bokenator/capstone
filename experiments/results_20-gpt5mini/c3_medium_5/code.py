# Complete implementation of compute_indicators and order_func for the MACD + ATR trailing stop strategy

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Attempt to import vectorbt enums; provide dynamic resolution/fallbacks
try:
    from vectorbt.portfolio.enums import SizeType, Direction
except Exception:  # pragma: no cover - safe fallback if enums not available at import time
    SizeType = None
    Direction = None


def _resolve_enum_member(enum_cls: Any, keywords: tuple, fallback: Any) -> Any:
    """
    Try to resolve an enum/class attribute that matches any of the provided keywords
    (case-insensitive substring match). If not found, return the fallback.

    This is defensive because different vectorbt versions expose enum members
    with slightly different names.
    """
    if enum_cls is None:
        return fallback

    # Inspect public attributes
    for attr in dir(enum_cls):
        if attr.startswith("_"):
            continue
        al = attr.lower()
        for kw in keywords:
            if kw in al:
                try:
                    return getattr(enum_cls, attr)
                except Exception:
                    continue
    # Try direct attribute access on common names
    for attr in keywords:
        if hasattr(enum_cls, attr):
            try:
                return getattr(enum_cls, attr)
            except Exception:
                pass
    return fallback


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dictionary with numpy arrays for: close, high, macd, signal, atr, sma.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least high/low/close)
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal EMA period
        sma_period: Period for trend filter SMA
        atr_period: ATR lookback period

    Returns:
        Dict[str, np.ndarray]
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    # Extract series and ensure float dtype
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd, macd_signal)
    # Use pandas ewm with adjust=False for standard EMA behaviour
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # ATR: True Range and simple moving average over atr_period
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Use simple rolling mean for ATR; set min_periods=1 to avoid NaNs early on
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    return {
        "close": close.values.astype(float),
        "high": high.values.astype(float),
        "macd": macd_line.values.astype(float),
        "signal": signal_line.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
    }


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[Any, Any, Any]:
    """
    Order function implementing the MACD crossover entries with ATR-based trailing stops.

    The function is designed to be called sequentially by vectorbt (use_numba=False).
    It keeps lightweight per-column state in function attributes and resets state at
    the start of each run (when c.i == 0).

    Returns a tuple (size, size_type, direction). If no action is required, size is
    set to np.nan (the harness will convert this to a no-order placeholder).

    Args:
        c: context object provided by vectorbt (must have at least 'i' and optionally 'col')
        close, high, macd, signal, atr, sma: numpy arrays of indicator values
        trailing_mult: multiplier for ATR to set trailing stop distance (e.g., 2.0)

    Returns:
        (size, size_type, direction)
    """
    # Resolve enum-like values (robust across vectorbt versions)
    size_amount = _resolve_enum_member(SizeType, ("amount", "amt"), 0)
    direction_both = _resolve_enum_member(Direction, ("both",), 0)

    # Safely get index and column from context
    i = int(getattr(c, "i", 0))
    col = getattr(c, "col", None)
    if col is None:
        # Some versions use 'column' attribute
        col = getattr(c, "column", 0)
    col = int(col)

    # Initialize per-column state container
    if not hasattr(order_func, "_state"):
        order_func._state = {}

    if col not in order_func._state or i == 0:
        # Reset state at the beginning of each run/column
        order_func._state[col] = {
            "in_pos": False,
            "entry_index": -1,
            "highest": -np.inf,
            "entry_size": 0.0,
        }

    state = order_func._state[col]

    # Helper to safely fetch current/previous values
    def _safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            v = float(arr[idx])
        except Exception:
            v = float("nan")
        return v

    curr_close = _safe_get(close, i)
    curr_high = _safe_get(high, i)
    curr_macd = _safe_get(macd, i)
    curr_signal = _safe_get(signal, i)
    curr_atr = _safe_get(atr, i)
    curr_sma = _safe_get(sma, i)

    prev_macd = _safe_get(macd, i - 1) if i > 0 else float("nan")
    prev_signal = _safe_get(signal, i - 1) if i > 0 else float("nan")

    # Detect MACD crosses (only use information up to current index)
    macd_cross_up = False
    macd_cross_down = False
    if not np.isnan(prev_macd) and not np.isnan(prev_signal) and not np.isnan(curr_macd) and not np.isnan(curr_signal):
        macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
        macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

    # Entry condition (long-only): MACD cross up AND price > SMA AND not already in position
    if (not state["in_pos"]) and macd_cross_up and (not np.isnan(curr_sma)) and (not np.isnan(curr_close)) and (curr_close > curr_sma):
        # Enter long: buy 1 unit. Store entry info to manage trailing stop
        state["in_pos"] = True
        state["entry_index"] = i
        state["highest"] = curr_high if not np.isnan(curr_high) else curr_close
        state["entry_size"] = 1.0
        return (1.0, size_amount, direction_both)

    # If in position, update highest and check exits
    if state["in_pos"]:
        # Update highest_since_entry using high of current bar (no lookahead)
        if not np.isnan(curr_high):
            if curr_high > state["highest"]:
                state["highest"] = curr_high

        # Exit on MACD bearish cross
        if macd_cross_down:
            # Close the position by selling the entry size (negative size with direction both)
            size_to_close = state.get("entry_size", 1.0) or 1.0
            # Reset state
            state["in_pos"] = False
            state["entry_index"] = -1
            state["highest"] = -np.inf
            state["entry_size"] = 0.0
            return (-abs(size_to_close), size_amount, direction_both)

        # Exit on trailing stop: close if close < highest_since_entry - trailing_mult * ATR
        if (not np.isnan(curr_atr)) and (not np.isnan(state["highest"])) and (not np.isnan(curr_close)):
            threshold = state["highest"] - trailing_mult * curr_atr
            if curr_close < threshold:
                size_to_close = state.get("entry_size", 1.0) or 1.0
                state["in_pos"] = False
                state["entry_index"] = -1
                state["highest"] = -np.inf
                state["entry_size"] = 0.0
                return (-abs(size_to_close), size_amount, direction_both)

    # No action
    return (np.nan, size_amount, direction_both)
