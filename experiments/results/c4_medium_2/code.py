import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


def order_func(
    c: object,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Order generation function for vectorbt.from_order_func (no numba).

    Implements a long-only MACD crossover entry with a 50-period SMA trend filter
    and an ATR-based trailing stop. Uses function attributes to keep simple state
    across calls (pending entry, entry index, highest price since entry).

    Returns order objects produced by vbt.portfolio.nb.order_nb for compatibility
    with vectorbt's simulation internals. Use vbt.portfolio.enums.NoOrder for no action.
    """
    # Initialize persistent state as attributes on the function object
    if not hasattr(order_func, "_state_initialized"):
        order_func._pending_entry = False  # True when an entry order has been emitted but position not yet observed
        order_func._entry_i = None         # Index when the last entry was initiated
        order_func._highest = np.nan       # Highest high seen since entry
        order_func._state_initialized = True

    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks for array bounds
    n = close.shape[0] if isinstance(close, np.ndarray) else 0
    if i < 0 or i >= n:
        # Out-of-bounds index: no action
        return vbt.portfolio.enums.NoOrder

    # Helper: check if values are finite
    def _is_finite(val) -> bool:
        try:
            return np.isfinite(val)
        except Exception:
            return False

    # Helper: MACD cross up
    def _macd_cross_up(idx: int) -> bool:
        if idx == 0:
            return False
        if not (_is_finite(macd[idx]) and _is_finite(signal[idx]) and _is_finite(macd[idx - 1]) and _is_finite(signal[idx - 1])):
            return False
        return (macd[idx - 1] < signal[idx - 1]) and (macd[idx] > signal[idx])

    # Helper: MACD cross down
    def _macd_cross_down(idx: int) -> bool:
        if idx == 0:
            return False
        if not (_is_finite(macd[idx]) and _is_finite(signal[idx]) and _is_finite(macd[idx - 1]) and _is_finite(signal[idx - 1])):
            return False
        return (macd[idx - 1] > signal[idx - 1]) and (macd[idx] < signal[idx])

    # If we observe an open position, ensure pending flag is cleared and initialize highest
    if pos > 0:
        # Entry was filled (or position exists from outside). Clear pending flag.
        if order_func._pending_entry:
            order_func._pending_entry = False
        # If we don't have an entry index recorded (e.g., strategy started already in a position), set it
        if order_func._entry_i is None:
            order_func._entry_i = i
        # Update highest since entry using today's high if available
        if _is_finite(high[i]):
            if not _is_finite(order_func._highest):
                order_func._highest = high[i]
            else:
                order_func._highest = float(np.maximum(order_func._highest, high[i]))

        # EXIT CONDITIONS (any):
        # 1) MACD crosses below signal
        if _macd_cross_down(i):
            # Reset state on exit
            order_func._entry_i = None
            order_func._highest = np.nan
            # Close entire long position: use percent size (-inf) and market execution (price=inf)
            return vbt.portfolio.nb.order_nb(-np.inf, np.inf, 2, 1)

        # 2) Price falls below (highest_since_entry - trailing_mult * ATR)
        if _is_finite(order_func._highest) and _is_finite(atr[i]) and _is_finite(close[i]):
            stop_level = order_func._highest - trailing_mult * atr[i]
            if close[i] < stop_level:
                order_func._entry_i = None
                order_func._highest = np.nan
                return vbt.portfolio.nb.order_nb(-np.inf, np.inf, 2, 1)

        # No exit triggered
        return vbt.portfolio.enums.NoOrder

    # NO POSITION: consider entry
    # If an entry was already emitted but not observed (pending), do nothing to avoid duplicate orders
    if order_func._pending_entry:
        return vbt.portfolio.enums.NoOrder

    # Entry condition: MACD cross up AND price above SMA
    if _macd_cross_up(i):
        if _is_finite(close[i]) and _is_finite(sma[i]):
            if close[i] > sma[i]:
                # Emit entry order: allocate 100% of equity to the long position
                order_func._pending_entry = True
                order_func._entry_i = i
                order_func._highest = high[i] if _is_finite(high[i]) else np.nan
                # Market order using percent of equity (100%)
                return vbt.portfolio.nb.order_nb(1.0, np.inf, 2, 1)

    # Default: no action
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
    Compute indicators required by the strategy using vectorbt indicator runners.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'. All values
    are numpy arrays matching the length of the input ohlcv.
    """
    # Validate required columns
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError('ohlcv DataFrame must contain "close" and "high" columns')

    # Use 'low' if available, otherwise fall back to 'close' to allow ATR computation
    low_series = ohlcv["low"] if "low" in ohlcv.columns else ohlcv["close"]

    # MACD
    macd_ind = vbt.MACD.run(ohlcv["close"], fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd
    signal_line = macd_ind.signal

    # ATR
    atr_ind = vbt.ATR.run(ohlcv["high"], low_series, ohlcv["close"], window=atr_period)
    atr_series = atr_ind.atr

    # SMA
    sma_ind = vbt.MA.run(ohlcv["close"], window=sma_period)
    sma_series = sma_ind.ma

    return {
        "close": ohlcv["close"].values,
        "high": ohlcv["high"].values,
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr_series.values,
        "sma": sma_series.values,
    }