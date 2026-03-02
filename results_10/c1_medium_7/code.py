import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Module-level state stored as a list to avoid string-keyed bracket access
# Index mapping:
# 0 -> in_position (bool)
# 1 -> entry_index (int or None)
# 2 -> highest_price (float)
_STATE = [False, None, np.nan]
_STATE_IN_POS = 0
_STATE_ENTRY_IDX = 1
_STATE_HIGHEST = 2


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
    Generate order at each bar using MACD crossover entries with ATR-based trailing stops.

    Entry:
    - MACD crosses above Signal AND price > SMA
    Exit:
    - MACD crosses below Signal OR price falls below (highest_since_entry - trailing_mult * ATR)

    Returns (size, size_type, direction) as required by the runner.
    """
    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, "position_now") else float(0.0)

    global _STATE

    # Reset or initialize state at the first bar
    if i == 0:
        _STATE[_STATE_IN_POS] = pos > 0
        _STATE[_STATE_ENTRY_IDX] = 0 if pos > 0 else None
        _STATE[_STATE_HIGHEST] = float(high[0]) if (pos > 0 and not np.isnan(high[0])) else np.nan

    # If a position exists but state was not set, initialize conservatively
    if pos > 0 and not _STATE[_STATE_IN_POS]:
        _STATE[_STATE_IN_POS] = True
        _STATE[_STATE_ENTRY_IDX] = i
        _STATE[_STATE_HIGHEST] = float(high[i]) if not np.isnan(high[i]) else float(close[i])

    def crossover_up(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = a[idx - 1], b[idx - 1]
        a_curr, b_curr = a[idx], b[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def crossover_down(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = a[idx - 1], b[idx - 1]
        a_curr, b_curr = a[idx], b[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    # No position -> consider entry
    if pos == 0:
        if i <= 0:
            return (np.nan, 0, 0)

        # Validate required indicator values exist
        if (
            np.isnan(macd[i])
            or np.isnan(signal[i])
            or np.isnan(macd[i - 1])
            or np.isnan(signal[i - 1])
            or np.isnan(sma[i])
            or np.isnan(close[i])
        ):
            return (np.nan, 0, 0)

        should_enter = crossover_up(macd, signal, i) and (close[i] > sma[i])

        if should_enter:
            # Initialize tracking state
            _STATE[_STATE_IN_POS] = True
            _STATE[_STATE_ENTRY_IDX] = i
            _STATE[_STATE_HIGHEST] = float(high[i]) if not np.isnan(high[i]) else float(close[i])

            # Enter with 100% equity (long-only)
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # Have position -> consider exit
    else:
        # Update highest price since entry
        if np.isnan(_STATE[_STATE_HIGHEST]):
            _STATE[_STATE_HIGHEST] = float(high[i]) if not np.isnan(high[i]) else float(close[i])
        else:
            if not np.isnan(high[i]):
                _STATE[_STATE_HIGHEST] = max(float(_STATE[_STATE_HIGHEST]), float(high[i]))

        atr_val = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        highest = _STATE[_STATE_HIGHEST]
        threshold = highest - trailing_mult * atr_val if (not np.isnan(highest) and not np.isnan(atr_val)) else np.nan

        macd_cross_down = crossover_down(macd, signal, i)
        price_broke_trail = (not np.isnan(threshold)) and (not np.isnan(close[i])) and (close[i] < threshold)

        if macd_cross_down or price_broke_trail:
            # Reset tracking state
            _STATE[_STATE_IN_POS] = False
            _STATE[_STATE_ENTRY_IDX] = None
            _STATE[_STATE_HIGHEST] = np.nan

            # Close entire long position
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
    Compute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    for col in ("close", "high"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column as per DATA_SCHEMA")

    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)

    # Use 'low' when available; otherwise fall back to close for ATR
    if "low" in ohlcv.columns:
        low_sr = ohlcv["low"].astype(float)
    else:
        low_sr = close_sr.copy()

    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    return {
        "close": close_sr.values,
        "high": high_sr.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }


__all__ = ["compute_indicators", "order_func"]
