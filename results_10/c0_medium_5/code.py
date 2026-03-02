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
    Compute MACD, Signal line, SMA and ATR indicators from OHLCV data.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...].
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for trend SMA.
        atr_period: Period for ATR.

    Returns:
        Dictionary with numpy arrays: macd, signal, atr, sma, close, high.

    Notes:
        - Uses pandas' ewm (adjust=False) for EMA/Wilder-style smoothing.
        - Aligns outputs to the input index; initial values will be NaN where
          indicators cannot be computed.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain column '{col}'")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD: EMA_fast - EMA_slow, signal = EMA(macd)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # ATR: True Range then Wilder's smoothing (EWMA with alpha=1/atr_period)
    prev_close = close.shift(1)
    high_low = high - low
    high_pc = (high - prev_close).abs()
    low_pc = (low - prev_close).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # SMA for trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # Return numpy arrays
    return {
        "macd": macd.values.astype(float),
        "signal": signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
        "close": close.values.astype(float),
        "high": high.values.astype(float),
    }


# Module-level fallback state store if OrderContext doesn't provide a mutable cache
_ORDER_STATE: Dict[int, Dict[str, Any]] = {}


def _get_state_store(c: Any) -> Dict[str, Any]:
    """
    Retrieve a mutable state store for the given order context.
    Prioritizes context.cache if available, otherwise falls back to a module-level dict
    keyed by id(c).
    """
    # Prefer a provided cache on the context (vectorbt exposes .cache for this purpose)
    if hasattr(c, "cache") and isinstance(getattr(c, "cache"), dict):
        return c.cache  # type: ignore[return-value]

    # Fallback: module-level store keyed by context id
    return _ORDER_STATE.setdefault(id(c), {})


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
    Order function to be used with vectorbt.Portfolio.from_order_func (use_numba=False).

    Logic:
    - Long entry when MACD crosses above signal AND price > 50-period SMA.
    - Exit when MACD crosses below OR price < (highest_since_entry - trailing_mult * ATR).
    - Trailing stop is based on the highest high seen since entry minus trailing_mult * ATR.

    Returns a tuple (size, size_type, direction):
    - size: float (use np.nan for no action)
    - size_type: int (enum value expected by vectorbt; we use Percent=1 by default)
    - direction: int (1 == Long)

    Important: This function does not use numba and returns plain Python tuples.
    """
    i = int(getattr(c, "i", 0))
    n = len(close) if close is not None else 0
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    state = _get_state_store(c)

    # Initialize persistent state in the store
    if "in_pos" not in state:
        state["in_pos"] = False
        state["entry_i"] = -1
        state["highest"] = float("-inf")
        state["stop_price"] = float("nan")

    # If indicators are NaN or we are at the first bar, do nothing
    if i == 0:
        return (np.nan, 0, 0)

    # Safely check for NaNs in required indicators at current index
    if np.isnan(macd[i]) or np.isnan(signal[i]) or np.isnan(sma[i]) or np.isnan(atr[i]):
        return (np.nan, 0, 0)

    # Define crossover helpers (safe against NaNs)
    def cross_above(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        return (not np.isnan(arr_a[idx - 1]) and not np.isnan(arr_b[idx - 1]) and
                not np.isnan(arr_a[idx]) and not np.isnan(arr_b[idx]) and
                arr_a[idx - 1] <= arr_b[idx - 1] and arr_a[idx] > arr_b[idx])

    def cross_below(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        return (not np.isnan(arr_a[idx - 1]) and not np.isnan(arr_b[idx - 1]) and
                not np.isnan(arr_a[idx]) and not np.isnan(arr_b[idx]) and
                arr_a[idx - 1] >= arr_b[idx - 1] and arr_a[idx] < arr_b[idx])

    price = float(close[i])
    high_price = float(high[i])

    macd_cross_up = cross_above(macd, signal, i)
    macd_cross_down = cross_below(macd, signal, i)

    # SizeType/Direction constants
    SIZE_TYPE_PERCENT = 1
    DIRECTION_LONG = 1

    # Entry logic
    if not state["in_pos"]:
        if macd_cross_up and price > sma[i]:
            # Enter: use 100% of available cash (Percent=1)
            state["in_pos"] = True
            state["entry_i"] = i
            state["highest"] = high_price
            state["stop_price"] = state["highest"] - trailing_mult * atr[i]
            return (1.0, SIZE_TYPE_PERCENT, DIRECTION_LONG)

        return (np.nan, 0, 0)

    # If in position, update highest and check exits
    if state["in_pos"]:
        # Update highest high since entry
        if high_price > state["highest"]:
            state["highest"] = high_price

        # Recompute trailing stop
        state["stop_price"] = state["highest"] - trailing_mult * atr[i]

        # Exit conditions: MACD crosses below OR price falls below trailing stop
        stop_hit = (not np.isnan(state["stop_price"])) and (price < state["stop_price"])
        if macd_cross_down or stop_hit:
            # Exit: set percent to 0
            state["in_pos"] = False
            state["entry_i"] = -1
            state["highest"] = float("-inf")
            state["stop_price"] = float("nan")
            return (0.0, SIZE_TYPE_PERCENT, DIRECTION_LONG)

        return (np.nan, 0, 0)

    return (np.nan, 0, 0)
