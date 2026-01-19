import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Returns a dict with keys: close, high, macd, signal, atr, sma.

    All returned values are numpy arrays aligned with the input ohlcv index.
    """
    # Validate input columns
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # Compute MACD
    macd_ind = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    macd_ser = macd_ind.macd
    signal_ser = macd_ind.signal

    # Compute ATR
    atr_ser = vbt.ATR.run(high, low, close, window=atr_period).atr

    # Compute SMA (trend filter)
    sma_ser = vbt.MA.run(close, window=sma_period).ma

    # Fill NaNs in a lookahead-free way (use past/constant fills only)
    # We replace remaining NaNs with 0.0 to ensure downstream code has finite values.
    macd_arr = macd_ser.fillna(0.0).values
    signal_arr = signal_ser.fillna(0.0).values
    atr_arr = atr_ser.fillna(0.0).values
    sma_arr = sma_ser.fillna(0.0).values

    close_arr = close.values
    high_arr = high.values

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }


def _extract_index_from_args(args: Tuple[Any, ...], n: int) -> int:
    # Try to find an integer index in args that fits the array length
    for a in args:
        if isinstance(a, int):
            ai = int(a)
            if 0 <= ai < n:
                return ai
    # Try to find object with common index attributes
    for a in args:
        if hasattr(a, "i"):
            try:
                ai = int(getattr(a, "i"))
                if 0 <= ai < n:
                    return ai
            except Exception:
                pass
        if hasattr(a, "index"):
            try:
                ai = int(getattr(a, "index"))
                if 0 <= ai < n:
                    return ai
            except Exception:
                pass
        if hasattr(a, "idx"):
            try:
                ai = int(getattr(a, "idx"))
                if 0 <= ai < n:
                    return ai
            except Exception:
                pass
    return -1


def _resolve_enum_values() -> Tuple[int, int, int]:
    """Resolve numeric enum values for Direction.LONG, Direction.CLOSE and SizeType.TargetPercent.

    The function attempts to introspect vbt.portfolio.enums and pick best-match names.
    Falls back to sensible defaults if resolution fails.
    """
    Direction = vbt.portfolio.enums.Direction
    SizeType = vbt.portfolio.enums.SizeType

    # Build mapping of attribute name -> int value
    dir_map = {}
    for name in dir(Direction):
        if name.startswith("_"):
            continue
        try:
            val = getattr(Direction, name)
            if isinstance(val, int):
                dir_map[name] = int(val)
            elif hasattr(val, "value"):
                try:
                    dir_map[name] = int(val.value)
                except Exception:
                    pass
        except Exception:
            pass

    size_map = {}
    for name in dir(SizeType):
        if name.startswith("_"):
            continue
        try:
            val = getattr(SizeType, name)
            if isinstance(val, int):
                size_map[name] = int(val)
            elif hasattr(val, "value"):
                try:
                    size_map[name] = int(val.value)
                except Exception:
                    pass
        except Exception:
            pass

    # Helper to find by keywords
    def find_by_keywords(m: dict, keywords: Tuple[str, ...]) -> int:
        for kw in keywords:
            for name, val in m.items():
                if kw.lower() in name.lower():
                    return val
        return None

    # Resolve LONG direction
    long_val = find_by_keywords(dir_map, ("LONG", "BUY", "OPEN"))
    if long_val is None:
        # fallback: any positive value
        for val in dir_map.values():
            if val > 0:
                long_val = val
                break
    if long_val is None:
        long_val = 1

    # Resolve CLOSE direction (prefer zero or names containing CLOSE/EXIT/SELL)
    close_val = find_by_keywords(dir_map, ("CLOSE", "EXIT", "SELL", "FLAT"))
    if close_val is None:
        # fallback to zero if present
        for val in dir_map.values():
            if val == 0:
                close_val = val
                break
    if close_val is None:
        # fallback to 0
        close_val = 0

    # Resolve SizeType.TargetPercent
    size_val = find_by_keywords(size_map, ("TARGETPERCENT", "TARGET_PERCENT", "TARGET", "PERCENT"))
    if size_val is None:
        # fallback to first available
        size_val = next(iter(size_map.values()), 1)

    return int(long_val), int(close_val), int(size_val)


def _make_order(direction: int, size_type: int, size: float):
    """Construct an order_nb object with the requested fields.

    Try multiple constructor signatures for order_nb until one works.
    We set price to np.inf to indicate a market order (the simulation code checks for np.isinf).
    """
    # Candidate constructor callables (try multiple signatures)
    cands = [
        lambda: vbt.portfolio.nb.order_nb(direction=int(direction), size_type=int(size_type), size=float(size), price=np.inf),
        lambda: vbt.portfolio.nb.order_nb(int(direction), int(size_type), float(size), np.inf),
        lambda: vbt.portfolio.nb.order_nb(int(direction), float(size), int(size_type), np.inf),
        lambda: vbt.portfolio.nb.order_nb(int(direction), np.inf, int(size_type), float(size)),
        lambda: vbt.portfolio.nb.order_nb(int(direction), float(size)),
        lambda: vbt.portfolio.nb.order_nb(),
    ]

    for cand in cands:
        try:
            ord_nb = cand()
            # If successful and has expected attributes, return
            if hasattr(ord_nb, "price") or hasattr(ord_nb, "direction") or hasattr(ord_nb, "size"):
                return ord_nb
        except Exception:
            continue

    # Fallback: return NoOrder sentinel
    return vbt.portfolio.enums.NoOrder


def order_func(*args: Any) -> Tuple[Any, ...]:
    """Order function compatible with vbt.Portfolio.from_order_func (use_numba=False).

    Robust parsing of arguments is implemented so the function works with vectorbt's
    Python-mode calling convention. The function maintains Python-level state on
    the function object to track open position and highest price since entry.
    """
    # Identify arrays (support both numpy arrays and pandas Series)
    arrays_all = [a for a in args if isinstance(a, (np.ndarray, pd.Series))]
    if len(arrays_all) < 6:
        # Not enough arrays provided -> no order
        return vbt.portfolio.enums.NoOrder

    # Take the last 6 arrays to map to the expected inputs (close, high, macd, signal, atr, sma)
    arrays = arrays_all[-6:]

    # Map arrays in the order they were provided to from_order_func
    close_arr = np.array(arrays[0])
    high_arr = np.array(arrays[1])
    macd_arr = np.array(arrays[2])
    signal_arr = np.array(arrays[3])
    atr_arr = np.array(arrays[4])
    sma_arr = np.array(arrays[5])

    n = len(close_arr)

    # Determine current index. Prefer explicit integer arg; fallback to call counter per array.
    index = _extract_index_from_args(args, n)

    # Initialize per-run counter/state if needed
    # Use array id to detect new run and reset state
    close_id = id(close_arr)
    if (not hasattr(order_func, "_last_close_id")) or (getattr(order_func, "_last_close_id") != close_id):
        order_func._last_close_id = close_id
        order_func._call_idx = 0
        order_func._in_position = False
        order_func._highest = -np.inf
        order_func._entry_index = -1
        # Resolve enum values once per run
        order_func._enum_values = _resolve_enum_values()

    if index == -1:
        # No explicit index found -> use internal call counter
        index = int(order_func._call_idx)
        order_func._call_idx += 1

    # Guard against out-of-bounds
    if index < 0 or index >= n:
        return vbt.portfolio.enums.NoOrder

    # Extract trailing_mult if provided (look for float-like args). Prefer the last float-like arg.
    float_args = [a for a in args if isinstance(a, float)]
    if len(float_args) > 0:
        trailing_mult = float(float_args[-1])
    else:
        # Default as per strategy description
        trailing_mult = 2.0

    # Current bar values
    curr_close = float(close_arr[index])
    curr_high = float(high_arr[index])
    curr_macd = float(macd_arr[index])
    curr_signal = float(signal_arr[index])
    curr_atr = float(atr_arr[index]) if index < len(atr_arr) else np.nan
    curr_sma = float(sma_arr[index])

    # Previous bar values for cross detection
    if index == 0:
        prev_macd = np.nan
        prev_signal = np.nan
    else:
        prev_macd = float(macd_arr[index - 1])
        prev_signal = float(signal_arr[index - 1])

    macd_cross_up = False
    macd_cross_down = False
    if index > 0:
        macd_cross_up = (curr_macd > curr_signal) and (prev_macd <= prev_signal)
        macd_cross_down = (curr_macd < curr_signal) and (prev_macd >= prev_signal)

    # Unpack enum numeric values
    long_dir, close_dir, size_type_target = order_func._enum_values

    # ENTRY: MACD bullish cross AND price above SMA
    if (not order_func._in_position) and macd_cross_up and (curr_close > curr_sma):
        order_func._in_position = True
        order_func._highest = curr_high
        order_func._entry_index = index
        return _make_order(long_dir, size_type_target, 1.0)

    # If in position, update highest_since_entry and check exits
    if order_func._in_position:
        if curr_high > order_func._highest:
            order_func._highest = curr_high

        # Trailing stop: price falls below highest_since_entry - trailing_mult * ATR
        if np.isfinite(curr_atr) and curr_atr > 0.0:
            stop_level = order_func._highest - (trailing_mult * curr_atr)
            if curr_close < stop_level:
                order_func._in_position = False
                return _make_order(close_dir, size_type_target, 0.0)

        # MACD bearish cross exit
        if macd_cross_down:
            order_func._in_position = False
            return _make_order(close_dir, size_type_target, 0.0)

    return vbt.portfolio.enums.NoOrder
