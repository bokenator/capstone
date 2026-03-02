import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Ensure numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    n = len(close)

    # Safety checks for index bounds
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    def is_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx-1]) or np.isnan(signal[idx-1]):
            return False
        return (macd[idx] > signal[idx]) and (macd[idx-1] <= signal[idx-1])

    def is_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx-1]) or np.isnan(signal[idx-1]):
            return False
        return (macd[idx] < signal[idx]) and (macd[idx-1] >= signal[idx-1])

    # Helper to robustly get entry index of the current open position
    def get_entry_index() -> int:
        # 1) Try c.pos_record_now.attribute access
        try:
            pr = c.pos_record_now
            try:
                val = pr.entry_idx
                if np.isfinite(val):
                    return int(val)
            except Exception:
                pass
        except Exception:
            pass

        # 2) Try c.last_lidx (may contain last long entry index)
        try:
            last_lidx = c.last_lidx
            try:
                return int(last_lidx)
            except Exception:
                try:
                    return int(last_lidx[0])
                except Exception:
                    pass
        except Exception:
            pass

        # 3) Fallback: scan backwards for the last MACD cross-up + price > sma
        for j in range(i, -1, -1):
            if j <= 0:
                continue
            try:
                if (not np.isnan(macd[j]) and not np.isnan(signal[j]) and
                    not np.isnan(macd[j-1]) and not np.isnan(signal[j-1]) and
                    (macd[j] > signal[j]) and (macd[j-1] <= signal[j-1]) and
                    (not np.isnan(sma[j])) and (close[j] > sma[j])):
                    return int(j)
            except Exception:
                continue

        # If nothing found, return 0 as conservative fallback
        return 0

    # ENTRY: when flat and MACD crosses above signal and price above SMA
    if pos == 0.0:
        if is_cross_up(i):
            # ensure SMA is available and price is above it
            if not np.isnan(sma[i]) and close[i] > sma[i]:
                # Buy with 50% of equity (Percent = 2)
                return (0.5, 2, 1)

    # EXIT: when in position
    else:
        # 1) MACD cross down -> exit
        if is_cross_down(i):
            return (-np.inf, 2, 1)  # Close entire long position

        # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        try:
            entry_idx = get_entry_index()
            # Ensure valid entry index
            if entry_idx is None:
                entry_idx = 0
            entry_idx = max(0, int(entry_idx))
            if entry_idx <= i:
                # Use high prices from entry to now to get the highest
                seg = high[entry_idx:i+1]
                if seg.size > 0:
                    highest = np.nanmax(seg)
                    atr_i = atr[i] if (i < len(atr)) else np.nan
                    if (not np.isnan(highest)) and (not np.isnan(atr_i)):
                        trailing_stop = highest - trailing_mult * float(atr_i)
                        # If current close falls below trailing stop -> exit
                        if not np.isnan(close[i]) and close[i] < trailing_stop:
                            return (-np.inf, 2, 1)
        except Exception:
            # On any error while computing trailing stop, do nothing (no action)
            pass

    # No action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators. Use vectorbt indicator classes.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are numpy arrays of the same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise KeyError("ohlcv must contain at least 'high' and 'close' columns")

    close_ser = ohlcv['close']
    high_ser = ohlcv['high']

    # low may be missing; fall back to close if so
    if 'low' in ohlcv.columns:
        low_ser = ohlcv['low']
    else:
        low_ser = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast,
                             slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    # Extract numpy arrays (ensure 1D)
    close_arr = close_ser.values.astype(float)
    high_arr = high_ser.values.astype(float)

    # macd_ind.macd and .signal might be pandas Series
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)

    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    # Ensure all arrays have same length
    length = len(close_arr)
    for arr in (high_arr, macd_arr, signal_arr, atr_arr, sma_arr):
        if len(arr) != length:
            raise ValueError("Indicator arrays length mismatch")

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }