import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


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

    Returns a dictionary with numpy arrays for keys:
    'close', 'high', 'macd', 'signal', 'atr', 'sma'

    All arrays are aligned with the input DataFrame index and have the
    same length as the input.
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Required columns
    for col in ("close", "high", "low"):
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv DataFrame must contain '{col}' column")

    close = ohlcv["close"].astype(float).copy()
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()

    # MACD: difference between fast and slow EMA, signal is EMA of MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR (True Range) with Wilder's smoothing (EMA with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Use min_periods=atr_period to produce NaNs until ATR warmup is ready
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    return {
        "close": close.to_numpy(),
        "high": high.to_numpy(),
        "macd": macd_line.to_numpy(),
        "signal": signal_line.to_numpy(),
        "atr": atr.to_numpy(),
        "sma": sma.to_numpy(),
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
) -> Tuple[float, int, int]:
    """
    Order function implementing the MACD + ATR trailing stop strategy.

    Args:
        c: context provided by vectorbt (has attribute `i` for current index) or
           an integer index when called outside of vectorbt in tests.
        close, high, macd, signal, atr, sma: 1-D numpy arrays of indicators
           aligned with the price series. Use only up to index `i` (no lookahead).
        trailing_mult: multiplier for ATR (e.g., 2.0)

    Returns:
        A tuple (size, size_type, direction):
          - size: float (np.nan for no order, otherwise amount or percent depending on size_type)
          - size_type: int (0 -> amount (units), 1 -> percent of portfolio)
          - direction: int (1 => long/buy, 0 => both (used for sells/closes))

    Notes:
        - This implementation uses amount sizing (size_type=0) with 1 unit for
          entries and exits to ensure exits close the opened position instead of opening a short.
        - Maintains minimal internal state on the function object between calls:
          position_open, entry_index, highest_since_entry, position_size.
        - Resets internal state when a new run starts (detect by index==0 or length change).
        - Synchronizes internal state with the simulator context if the context exposes
          current position size to avoid assuming fills.
    """
    import numpy as _np

    # Resolve index 'i' from context 'c' (support both vectorbt context and plain int)
    if hasattr(c, "i"):
        try:
            i = int(c.i)
        except Exception:
            i = 0
    elif isinstance(c, (int, _np.integer)):
        i = int(c)
    else:
        # Fallback: try attribute named 'index'
        idx = getattr(c, "index", None)
        try:
            i = int(idx) if idx is not None else 0
        except Exception:
            i = 0

    # Initialize or reset persistent state at start of run or when length changes
    state = getattr(order_func, "_state", None)
    length = len(close) if hasattr(close, "__len__") else None
    if state is None or state.get("last_len") != length or i == 0:
        order_func._state = {
            "position_open": False,
            "entry_index": None,
            "highest_since_entry": _np.nan,
            "position_size": 0.0,
            "last_len": length,
        }
    state = order_func._state

    # Try to synchronize with actual position size exposed by the simulator context
    def _get_pos_from_context(ctx: Any):
        # Common attribute names that may contain current position size
        candidates = [
            "position_size",
            "position",
            "pos",
            "size",
            "current_position",
            "current_size",
            "prev_size",
        ]
        for name in candidates:
            if hasattr(ctx, name):
                try:
                    val = getattr(ctx, name)
                    # If it's array-like, attempt to extract scalar
                    if hasattr(val, "item"):
                        val = val.item()
                    return float(val)
                except Exception:
                    try:
                        # Try converting to float directly
                        return float(getattr(ctx, name))
                    except Exception:
                        continue
        return None

    pos_from_ctx = _get_pos_from_context(c)
    if pos_from_ctx is not None:
        try:
            if pos_from_ctx > 0:
                state["position_open"] = True
                state["position_size"] = float(pos_from_ctx)
            else:
                state["position_open"] = False
                state["position_size"] = 0.0
        except Exception:
            pass

    # Helper to safely get value at index i
    def safe_get(arr, idx):
        try:
            val = arr[idx]
            # Convert numpy types to python float
            return float(val) if not _np.isnan(val) else _np.nan
        except Exception:
            return _np.nan

    curr_close = safe_get(close, i)
    curr_high = safe_get(high, i)
    macd_i = safe_get(macd, i)
    signal_i = safe_get(signal, i)
    atr_i = safe_get(atr, i)
    sma_i = safe_get(sma, i)

    # Default: no order
    no_order = (_np.nan, 0, 0)

    pos_open = bool(state.get("position_open", False))

    # Determine MACD cross signals using only past data (i and i-1)
    macd_cross_up = False
    macd_cross_down = False
    if i > 0:
        prev_macd = safe_get(macd, i - 1)
        prev_signal = safe_get(signal, i - 1)
        if (
            not _np.isnan(prev_macd)
            and not _np.isnan(prev_signal)
            and not _np.isnan(macd_i)
            and not _np.isnan(signal_i)
        ):
            macd_cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)
            macd_cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

    # ENTRY condition: MACD crosses up and price above SMA
    if (not pos_open) and macd_cross_up and (not _np.isnan(sma_i)) and (not _np.isnan(curr_close)) and (curr_close > sma_i):
        # Enter long using 1 unit (size_type=0 -> amount). Do NOT assume fill; rely on context sync.
        size = 1.0
        size_type = 0  # amount in units
        direction = 1  # long
        return (size, size_type, direction)

    # If in position, evaluate trailing stop and MACD bearish cross for exit
    if pos_open:
        # Update highest_since_entry
        prev_high = state.get("highest_since_entry", _np.nan)
        if _np.isnan(prev_high):
            state["highest_since_entry"] = curr_high
        else:
            if not _np.isnan(curr_high) and curr_high > prev_high:
                state["highest_since_entry"] = curr_high

        highest = state.get("highest_since_entry", _np.nan)
        # Trailing stop check (use ATR at current bar)
        if not _np.isnan(highest) and not _np.isnan(atr_i) and not _np.isnan(curr_close):
            trailing_stop = highest - float(trailing_mult) * atr_i
            if curr_close < trailing_stop:
                # Exit all: sell the same amount we think we hold (amount sizing)
                size = float(state.get("position_size", 1.0))
                size_type = 0
                direction = 0  # Both (allow sell to close)
                return (size, size_type, direction)

        # MACD bearish cross exit
        if macd_cross_down:
            size = float(state.get("position_size", 1.0))
            size_type = 0
            direction = 0
            return (size, size_type, direction)

    return no_order
