import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Optional


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute required indicators for the strategy.

    Returns a dict with numpy arrays for keys:
      - close, high, macd, signal, atr, sma

    Args:
        ohlcv: DataFrame containing at least ['high','low','close'] columns.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    close = ohlcv["close"].fillna(method="ffill").fillna(method="bfill")
    high = ohlcv["high"].fillna(method="ffill").fillna(method="bfill")
    low = ohlcv["low"].fillna(method="ffill").fillna(method="bfill")

    # MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd = macd_ind.macd
    signal = macd_ind.signal

    # SMA (trend filter)
    sma_ind = vbt.MA.run(close, window=sma_period)
    sma = sma_ind.ma

    # ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    atr = atr_ind.atr

    # Convert to numpy arrays and ensure lengths match
    result = {
        "close": close.values,
        "high": high.values,
        "macd": np.array(macd.values, dtype=np.float64),
        "signal": np.array(signal.values, dtype=np.float64),
        "atr": np.array(atr.values, dtype=np.float64),
        "sma": np.array(sma.values, dtype=np.float64),
    }

    return result


def order_func(order_ctx: Any, price: Any, close: Any = None, high: Any = None, macd: Any = None, signal: Any = None, atr: Any = None, sma: Any = None, trailing_mult: float = 2.0):
    """
    Stateful order function for vectorbt.from_order_func (use_numba=False).

    This implementation computes orders on the fly using closure state that persists
    across sequential calls. It adheres to the strategy logic:
      - Entry: MACD crosses above Signal AND price > SMA
      - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns either vbt.portfolio.enums.NoOrder or a vbt.portfolio.nb.order_nb(...) object
    for compatibility with vectorbt's Python simulation.
    """
    # Convert inputs (they might be numpy arrays or pandas Series)
    if close is None:
        raise ValueError("order_func requires `close` array argument")

    close_arr = np.array(close)
    high_arr = np.array(high) if high is not None else close_arr
    macd_arr = np.array(macd) if macd is not None else np.full(len(close_arr), np.nan, dtype=float)
    signal_arr = np.array(signal) if signal is not None else np.full(len(close_arr), np.nan, dtype=float)
    atr_arr = np.array(atr) if atr is not None else np.full(len(close_arr), np.nan, dtype=float)
    sma_arr = np.array(sma) if sma is not None else np.full(len(close_arr), np.nan, dtype=float)

    # Build a cache key and initialize state per unique input
    cache_key = (id(close_arr), id(high_arr), id(macd_arr), id(signal_arr), id(atr_arr), id(sma_arr), float(trailing_mult))
    if not hasattr(order_func, "_state") or order_func._state.get("cache_key") != cache_key:
        order_func._state = {
            "cache_key": cache_key,
            "in_position": False,
            "highest_since_entry": -np.inf,
            "last_idx": -1,
        }

    state = order_func._state

    # Extract current index from order_ctx
    if hasattr(order_ctx, "i"):
        idx = int(order_ctx.i)
    elif hasattr(order_ctx, "index"):
        idx = int(order_ctx.index)
    else:
        try:
            idx = int(order_ctx)
        except Exception:
            raise TypeError("Unable to extract integer index from order_ctx")

    # Ensure sequential operation: reset if idx goes backwards
    if idx <= state.get("last_idx", -1):
        state["in_position"] = False
        state["highest_since_entry"] = -np.inf

    state["last_idx"] = idx

    # Determine if MACD cross up/down at this index
    macd_cross_up = False
    macd_cross_down = False
    if idx >= 1:
        if np.isfinite(macd_arr[idx]) and np.isfinite(signal_arr[idx]) and np.isfinite(macd_arr[idx - 1]) and np.isfinite(signal_arr[idx - 1]):
            macd_cross_up = (macd_arr[idx] > signal_arr[idx]) and (macd_arr[idx - 1] <= signal_arr[idx - 1])
            macd_cross_down = (macd_arr[idx] < signal_arr[idx]) and (macd_arr[idx - 1] >= signal_arr[idx - 1])

    price = close_arr[idx]

    # Use NoOrder sentinel as default
    no_order = vbt.portfolio.enums.NoOrder

    # Attempt to pick SizeType and Direction members dynamically
    SizeType = getattr(vbt.portfolio.enums, "SizeType", None)
    Direction = getattr(vbt.portfolio.enums, "Direction", None)

    size_type_member = None
    direction_member = None

    try:
        if SizeType is not None and hasattr(SizeType, "__members__"):
            st_names = list(SizeType.__members__.keys())
            candidates = [name for name in st_names if any(x in name.upper() for x in ["TARGET", "PCT", "PERCENT"]) ]
            if candidates:
                size_type_member = SizeType.__members__[candidates[0]]
            else:
                size_type_member = SizeType.__members__[st_names[0]]
    except Exception:
        size_type_member = None

    try:
        if Direction is not None and hasattr(Direction, "__members__"):
            d_names = list(Direction.__members__.keys())
            d_candidates = [name for name in d_names if "LONG" in name.upper()]
            if d_candidates:
                direction_member = Direction.__members__[d_candidates[0]]
            else:
                direction_member = Direction.__members__[d_names[0]]
    except Exception:
        direction_member = None

    if size_type_member is None:
        size_type_member = 2
    if direction_member is None:
        direction_member = 1

    # Entry condition
    if not state["in_position"]:
        if macd_cross_up and np.isfinite(sma_arr[idx]) and np.isfinite(price) and (price > sma_arr[idx]):
            # Enter: create order object via nb.order_nb if possible
            try:
                order = vbt.portfolio.nb.order_nb(1.0, size_type_member, direction_member)
            except Exception:
                order = (1.0, size_type_member, direction_member)
            state["in_position"] = True
            state["highest_since_entry"] = high_arr[idx] if np.isfinite(high_arr[idx]) else price
            return order
        else:
            return no_order

    # If in position, update highest and check exit conditions
    if state["in_position"]:
        if np.isfinite(high_arr[idx]) and (high_arr[idx] > state["highest_since_entry"]):
            state["highest_since_entry"] = high_arr[idx]

        atr_t = atr_arr[idx] if idx < len(atr_arr) else np.nan
        trailing_level = np.nan
        if np.isfinite(state["highest_since_entry"]) and np.isfinite(atr_t):
            trailing_level = state["highest_since_entry"] - (trailing_mult * atr_t)

        if macd_cross_down:
            try:
                order = vbt.portfolio.nb.order_nb(0.0, size_type_member, direction_member)
            except Exception:
                order = (0.0, size_type_member, direction_member)
            state["in_position"] = False
            state["highest_since_entry"] = -np.inf
            return order

        if np.isfinite(price) and np.isfinite(trailing_level) and (price < trailing_level):
            try:
                order = vbt.portfolio.nb.order_nb(0.0, size_type_member, direction_member)
            except Exception:
                order = (0.0, size_type_member, direction_member)
            state["in_position"] = False
            state["highest_since_entry"] = -np.inf
            return order

    return no_order
