import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

# Global state store keyed by id(context) to persist state across order_func calls
_CTX_STATE: Dict[int, Dict[str, Any]] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy:
      - MACD line and Signal line (EMA-based)
      - SMA for trend filter
      - ATR for trailing stop

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close'] (case-insensitive)
        macd_fast: fast EMA period for MACD
        macd_slow: slow EMA period for MACD
        macd_signal: signal EMA period for MACD
        sma_period: period for simple moving average trend filter
        atr_period: period for ATR

    Returns:
        Dictionary of numpy arrays with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'

    Notes:
        - All returned arrays have the same length as the input ohlcv.
        - Uses only past and current data (no lookahead).
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    col_map = {col.lower(): col for col in ohlcv.columns}
    required = ["open", "high", "low", "close"]
    for r in required:
        if r not in col_map:
            raise KeyError(f"ohlcv must contain '{r}' column (case-insensitive)")

    # Extract series
    close = ohlcv[col_map["close"]].astype(float)
    high = ohlcv[col_map["high"]].astype(float)
    low = ohlcv[col_map["low"]].astype(float)
    open_s = ohlcv[col_map["open"]].astype(float)

    # MACD (EMA) - use pandas ewm with adjust=False (standard recursive EMA)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR (True Range then rolling mean). Use simple moving average for ATR to keep it clear.
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # Convert to numpy arrays
    return {
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr.values,
        "sma": sma.values,
        "close": close.values,
        "high": high.values,
    }


def _get_ctx_state(c: Any) -> Dict[str, Any]:
    """Retrieve or create persistent state dict for a given context object."""
    key = id(c)
    if key not in _CTX_STATE:
        _CTX_STATE[key] = {
            "in_pos": False,
            "entry_i": -1,
            "highest": -np.inf,
            "entry_price": np.nan,
        }
    return _CTX_STATE[key]


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
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Returns a tuple (size, size_type, direction):
      - size: float (np.nan for no order)
      - size_type: int (encoded enum value; using Amount = 0)
      - direction: int (encoded enum value; using Long = 1, Short = 2)

    Implementation details / assumptions:
      - We use SizeType.Amount (int 0) and Direction.Long/Short (1/2).
        Entry: buy 1 unit. Exit: sell 1 unit.
      - State is stored in a global dict keyed by id(c).
      - All decisions use only data up to index c.i to avoid lookahead.
    """
    i = int(getattr(c, "i", 0))

    state = _get_ctx_state(c)

    # Helper enums (use ints to avoid importing vectorbt enums)
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    # Safety: if any required indicator is NaN at i, do not place orders
    def is_nan_at(arr: np.ndarray, idx: int) -> bool:
        try:
            return np.isnan(arr[idx])
        except Exception:
            return True

    if is_nan_at(macd, i) or is_nan_at(signal, i) or is_nan_at(atr, i) or is_nan_at(sma, i):
        return (np.nan, 0, 0)

    price = float(close[i])
    cur_macd = float(macd[i])
    cur_signal = float(signal[i])
    cur_atr = float(atr[i])
    cur_sma = float(sma[i])
    cur_high = float(high[i])

    # Previous values (for crossover detection). If missing, treat as no crossover.
    if i > 0 and not (np.isnan(macd[i - 1]) or np.isnan(signal[i - 1])):
        prev_macd = float(macd[i - 1])
        prev_signal = float(signal[i - 1])
    else:
        prev_macd = np.nan
        prev_signal = np.nan

    macd_cross_up = False
    macd_cross_down = False

    if not np.isnan(prev_macd) and not np.isnan(prev_signal):
        macd_cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
        macd_cross_down = (prev_macd >= prev_signal) and (cur_macd < cur_signal)

    # ENTRY: MACD crosses above signal AND price > SMA
    if not state["in_pos"]:
        if macd_cross_up and price > cur_sma:
            # Enter long: buy 1 unit
            state["in_pos"] = True
            state["entry_i"] = i
            state["highest"] = cur_high
            state["entry_price"] = price
            return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
        else:
            return (np.nan, 0, 0)

    # If in position, update highest_since_entry
    if cur_high > state["highest"]:
        state["highest"] = cur_high

    # Compute trailing stop price using highest since entry
    trailing_stop_price = state["highest"] - trailing_mult * cur_atr

    # EXIT: MACD crosses below OR price falls below trailing stop
    if macd_cross_down or (price < trailing_stop_price):
        # Exit by selling 1 unit
        state["in_pos"] = False
        state["entry_i"] = -1
        state["highest"] = -np.inf
        state["entry_price"] = np.nan
        return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)

    # Otherwise, no order
    return (np.nan, 0, 0)
