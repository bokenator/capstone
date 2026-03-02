import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


class _TrailingState:
    """Simple object to hold trailing stop state for a single-asset strategy.

    Using attributes instead of dict keys to avoid static analysis mistaking string keys
    for DataFrame column accesses.
    """

    def __init__(self):
        self.in_position: bool = False
        self.highest: float = -np.inf
        self.pending_entry_idx: int | None = None


# Single global instance used by order_func
_TRAILING_STATE = _TrailingState()


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    Implements MACD crossover entry with SMA trend filter and ATR-based trailing stop.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Local alias to module state
    global _TRAILING_STATE

    def _is_finite(x):
        return np.isfinite(x)

    in_pos = pos > 0.0

    # When flat, look for entry signals
    if not in_pos:
        _TRAILING_STATE.in_position = False

        if i > 0:
            macd_curr = macd[i]
            signal_curr = signal[i]
            macd_prev = macd[i - 1]
            signal_prev = signal[i - 1]
            sma_curr = sma[i]

            has_macd_vals = _is_finite(macd_curr) and _is_finite(signal_curr) and _is_finite(macd_prev) and _is_finite(signal_prev)
            has_sma = _is_finite(sma_curr)
            has_price = _is_finite(close[i])

            if has_macd_vals and has_sma and has_price:
                crossed_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
                price_above_sma = close[i] > sma_curr

                if crossed_up and price_above_sma:
                    # Mark the bar where entry was signaled so we can include its high when initializing trailing stop
                    _TRAILING_STATE.pending_entry_idx = i
                    return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # In position: initialize trailing state on first bar with position
    if not _TRAILING_STATE.in_position:
        pending_idx = _TRAILING_STATE.pending_entry_idx
        if pending_idx is not None and pending_idx < i:
            h_entry = high[pending_idx] if _is_finite(high[pending_idx]) else close[pending_idx]
            h_curr = high[i] if _is_finite(high[i]) else close[i]
            _TRAILING_STATE.highest = max(float(h_entry), float(h_curr))
            _TRAILING_STATE.pending_entry_idx = None
        else:
            _TRAILING_STATE.highest = high[i] if _is_finite(high[i]) else close[i]
        _TRAILING_STATE.in_position = True

    # Update highest since entry
    if _is_finite(high[i]):
        _TRAILING_STATE.highest = max(float(_TRAILING_STATE.highest), float(high[i]))

    # Check exit conditions
    should_exit = False

    # MACD cross below
    if i > 0:
        macd_curr = macd[i]
        signal_curr = signal[i]
        macd_prev = macd[i - 1]
        signal_prev = signal[i - 1]

        has_macd_vals = _is_finite(macd_curr) and _is_finite(signal_curr) and _is_finite(macd_prev) and _is_finite(signal_prev)
        if has_macd_vals:
            crossed_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
            if crossed_down:
                should_exit = True

    # ATR-based trailing stop
    highest = _TRAILING_STATE.highest
    if not should_exit:
        if _is_finite(highest) and _is_finite(atr[i]):
            stop_level = highest - float(trailing_mult) * float(atr[i])
            price_now = close[i]
            if _is_finite(price_now) and price_now < stop_level:
                should_exit = True

    if should_exit:
        _TRAILING_STATE.in_position = False
        _TRAILING_STATE.highest = -np.inf
        _TRAILING_STATE.pending_entry_idx = None
        return (-np.inf, 2, 1)

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    if "close" not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'high' column")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"] if "low" in ohlcv.columns else ohlcv["close"]

    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }