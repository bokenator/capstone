import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Fallback state storage when OrderContext doesn't allow attribute assignment.
_ORDER_CTX_STATE: Dict[int, Dict[object, float]] = {}
# Use a non-string key to avoid being mistaken for a DataFrame column in static analysis
_ENTRY_KEY = ("__vbt_entry_highest__",)


def _get_ctx_state(c: Any) -> Dict[object, float]:
    """Return a mutable state mapping for the given OrderContext.

    Prefer using c.cache if available (vectorbt may provide a cache dict).
    Otherwise use a module-level dict keyed by id(c).
    """
    # Try to use built-in cache if present
    try:
        cache = getattr(c, "cache", None)
    except Exception:
        cache = None

    if isinstance(cache, dict):
        return cache

    # Fallback to module-level storage keyed by id(c)
    ctx_id = id(c)
    if ctx_id not in _ORDER_CTX_STATE:
        _ORDER_CTX_STATE[ctx_id] = {}
    return _ORDER_CTX_STATE[ctx_id]


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
    Order function for vectorbt.from_order_func implementing MACD entries
    with ATR-based trailing stops.

    Logic:
    - Entry when MACD crosses above Signal AND close > SMA
    - Exit when MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    State is persisted in a context-specific mapping (c.cache if available,
    otherwise a module-level fallback keyed by id(c)).

    Returns:
        An order object created by vbt.portfolio.nb.order_nb OR vbt.portfolio.enums.NoOrder for no action.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Access persistent state dict for this context
    _state = _get_ctx_state(c)

    # Helper to safely access array values
    def _val(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return float(np.nan)

    # Current values
    close_i = _val(close, i)
    high_i = _val(high, i)
    macd_i = _val(macd, i)
    signal_i = _val(signal, i)
    atr_i = _val(atr, i)
    sma_i = _val(sma, i)

    # No action if we don't have enough data for comparisons
    if i == 0:
        return vbt.portfolio.enums.NoOrder

    # Previous values for crossover detection
    prev_macd = _val(macd, i - 1)
    prev_signal = _val(signal, i - 1)

    # ENTRY: Only when flat (no position)
    if pos == 0.0:
        # Reset any stale state
        if _ENTRY_KEY in _state:
            try:
                del _state[_ENTRY_KEY]
            except Exception:
                _state[_ENTRY_KEY] = None

        # Require all necessary finite values for entry check
        if (
            np.isfinite(prev_macd)
            and np.isfinite(prev_signal)
            and np.isfinite(macd_i)
            and np.isfinite(signal_i)
            and np.isfinite(sma_i)
            and np.isfinite(close_i)
        ):
            macd_cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)
            price_above_sma = close_i > sma_i

            if macd_cross_up and price_above_sma:
                # Initialize highest price since entry with today's high (fallback to close)
                _state[_ENTRY_KEY] = high_i if np.isfinite(high_i) else close_i
                # Enter full equity long (100% of equity)
                return vbt.portfolio.nb.order_nb(
                    1.0,
                    vbt.portfolio.enums.SizeType.Percent,
                    vbt.portfolio.enums.Direction.LongOnly,
                )

        return vbt.portfolio.enums.NoOrder

    # IN POSITION: Track highest since entry and check exits
    else:
        entry_highest = _state.get(_ENTRY_KEY, None)

        # Update highest_since_entry
        if entry_highest is None or not np.isfinite(entry_highest):
            entry_highest = high_i if np.isfinite(high_i) else close_i
        else:
            if np.isfinite(high_i):
                entry_highest = float(max(entry_highest, high_i))

        _state[_ENTRY_KEY] = entry_highest

        # Compute trailing stop price if ATR is available
        stop_price = None
        if np.isfinite(atr_i) and np.isfinite(entry_highest):
            stop_price = entry_highest - float(trailing_mult) * atr_i

        # MACD bearish cross detection
        macd_cross_down = False
        if np.isfinite(prev_macd) and np.isfinite(prev_signal) and np.isfinite(macd_i) and np.isfinite(signal_i):
            macd_cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

        # Price-based trailing stop trigger
        price_below_stop = False
        if stop_price is not None and np.isfinite(close_i):
            price_below_stop = close_i < stop_price

        if macd_cross_down or price_below_stop:
            # Clear stored state
            try:
                del _state[_ENTRY_KEY]
            except Exception:
                _state[_ENTRY_KEY] = None
            # Close entire long position
            return vbt.portfolio.nb.order_nb(
                -np.inf,
                vbt.portfolio.enums.SizeType.Percent,
                vbt.portfolio.enums.Direction.LongOnly,
            )

        return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy using vectorbt.

    Returns a dictionary with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if "close" not in ohlcv:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv:
        raise ValueError("ohlcv must contain 'high' column")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]
    low_series = ohlcv["low"] if "low" in ohlcv else close_series

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd.values
    signal_line = macd_ind.signal.values

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_line = atr_ind.atr.values

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_line = sma_ind.ma.values

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": np.array(macd_line, dtype=float),
        "signal": np.array(signal_line, dtype=float),
        "atr": np.array(atr_line, dtype=float),
        "sma": np.array(sma_line, dtype=float),
    }