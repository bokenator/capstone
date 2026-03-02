import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


class _State:
    """Simple container to hold order-function state without using string keys.

    Using attribute access avoids creating arbitrary string literals that may be
    mistaken for DataFrame column access by static analyzers.
    """

    def __init__(self):
        self.in_pos: bool = False
        self.highest: float = np.nan
        self.entry_idx: int | None = None


# Module-level state (one object reused across calls). Reset at the start of each
# simulation when order_func detects c.i == 0.
_ORDER_STATE = _State()


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
    Order function implementing MACD crossover entries with ATR trailing stops.

    Returns a tuple (size, size_type, direction) as required by the runner.
    """
    global _ORDER_STATE

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the start of a simulation
    if i == 0:
        _ORDER_STATE.in_pos = False
        _ORDER_STATE.highest = np.nan
        _ORDER_STATE.entry_idx = None

    # Helper to validate indices
    def is_valid_index(idx: int) -> bool:
        return 0 <= idx < len(close)

    def is_cross_up(idx: int) -> bool:
        if not is_valid_index(idx) or idx == 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx] > signal[idx]) and (macd[idx - 1] <= signal[idx - 1])

    def is_cross_down(idx: int) -> bool:
        if not is_valid_index(idx) or idx == 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx] < signal[idx]) and (macd[idx - 1] >= signal[idx - 1])

    # ENTRY logic
    if pos == 0.0 and (not _ORDER_STATE.in_pos):
        # Trend filter: price above SMA
        price_above_sma = False
        if is_valid_index(i) and not np.isnan(close[i]) and not np.isnan(sma[i]):
            price_above_sma = close[i] > sma[i]

        # MACD bullish cross
        macd_cross_up = is_cross_up(i)

        if macd_cross_up and price_above_sma:
            # Mark state as in position from this point
            _ORDER_STATE.in_pos = True
            _ORDER_STATE.entry_idx = i
            _ORDER_STATE.highest = high[i] if (is_valid_index(i) and not np.isnan(high[i])) else close[i]

            # Enter: use 50% of equity (SizeType.Percent -> 2)
            return (0.5, 2, 1)

    # POSITION MANAGEMENT
    if pos != 0.0 or _ORDER_STATE.in_pos:
        # Update highest since entry
        if is_valid_index(i) and not np.isnan(high[i]):
            if np.isnan(_ORDER_STATE.highest):
                _ORDER_STATE.highest = high[i]
            else:
                if high[i] > _ORDER_STATE.highest:
                    _ORDER_STATE.highest = high[i]

        should_exit = False

        # Exit on MACD bearish cross
        if is_cross_down(i):
            should_exit = True

        # Trailing stop: price falls below highest_since_entry - trailing_mult * ATR
        stop_trigger = False
        if is_valid_index(i) and not np.isnan(atr[i]) and not np.isnan(_ORDER_STATE.highest):
            stop_price = _ORDER_STATE.highest - trailing_mult * atr[i]
            if np.isfinite(stop_price) and not np.isnan(close[i]):
                if close[i] < stop_price:
                    stop_trigger = True

        if stop_trigger:
            should_exit = True

        if should_exit:
            # Reset state
            _ORDER_STATE.in_pos = False
            _ORDER_STATE.highest = np.nan
            _ORDER_STATE.entry_idx = None

            # Close entire long position using -inf percent signal
            return (-np.inf, 2, 1)

    # No action
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
    Compute indicators required by the strategy using vectorbt.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """

    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_sr = ohlcv['close'].astype(float)
    high_sr = ohlcv['high'].astype(float)

    # Low is optional in DATA_SCHEMA; fallback to close if not provided
    if 'low' in ohlcv.columns:
        low_sr = ohlcv['low'].astype(float)
    else:
        low_sr = close_sr.copy()

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }