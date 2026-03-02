import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Order function for vectorbt.from_order_func (pure Python, NO NUMBA).

    Logic:
    - Entry: when MACD crosses above Signal AND price > 50-SMA -> enter long with 50% of equity
    - Exit: when MACD crosses below Signal OR price falls below (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction) as specified in the prompt.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks: ensure index within arrays
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Helper: safe element access
    def val(arr: np.ndarray, idx: int):
        try:
            return arr[idx]
        except Exception:
            return np.nan

    # Helpers to detect crossovers at index i
    def is_cross_above(idx: int) -> bool:
        if idx <= 0:
            return False
        a = val(macd, idx)
        b = val(signal, idx)
        a_prev = val(macd, idx - 1)
        b_prev = val(signal, idx - 1)
        if np.isnan(a) or np.isnan(b) or np.isnan(a_prev) or np.isnan(b_prev):
            return False
        return (a > b) and (a_prev <= b_prev)

    def is_cross_below(idx: int) -> bool:
        if idx <= 0:
            return False
        a = val(macd, idx)
        b = val(signal, idx)
        a_prev = val(macd, idx - 1)
        b_prev = val(signal, idx - 1)
        if np.isnan(a) or np.isnan(b) or np.isnan(a_prev) or np.isnan(b_prev):
            return False
        return (a < b) and (a_prev >= b_prev)

    # ENTRY: flat -> check for MACD cross above + price above SMA
    if pos == 0.0:
        if is_cross_above(i):
            c_close = val(close, i)
            c_sma = val(sma, i)
            if not np.isnan(c_close) and not np.isnan(c_sma) and (c_close > c_sma):
                # Enter long with 50% of equity
                return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # If here, we have a position - check for exits
    # 1) MACD cross below -> close position
    if is_cross_below(i):
        return (-np.inf, 2, 1)

    # 2) ATR-based trailing stop from highest price since entry
    # Find the most recent valid entry index (last cross above where price > sma)
    entry_idx = None
    # Search backwards for last cross above event
    for j in range(i, 0, -1):
        if is_cross_above(j):
            j_close = val(close, j)
            j_sma = val(sma, j)
            if not np.isnan(j_close) and not np.isnan(j_sma) and (j_close > j_sma):
                entry_idx = j
                break

    if entry_idx is None:
        # Could not determine entry index reliably; do not exit on trailing stop
        return (np.nan, 0, 0)

    # Compute highest high since entry (inclusive)
    try:
        highs_segment = high[entry_idx : i + 1]
        # Use nanmax to ignore NaNs; if all NaNs, nanmax will raise, so catch
        highest_since_entry = np.nanmax(highs_segment)
    except Exception:
        highest_since_entry = np.nan

    current_atr = val(atr, i)

    if np.isnan(highest_since_entry) or np.isnan(current_atr):
        return (np.nan, 0, 0)

    trailing_stop_level = highest_since_entry - float(trailing_mult) * float(current_atr)

    # If current close falls below trailing stop level -> exit
    if not np.isnan(val(close, i)) and (val(close, i) < trailing_stop_level):
        return (-np.inf, 2, 1)

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy using vectorbt indicators.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy array of the same length as the input ohlcv.
    """
    # Validate required columns per DATA_SCHEMA
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # MACD
    macd_ind = vbt.MACD.run(
        close_series,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }