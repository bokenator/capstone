import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Global variable to track the most recent entry index for trailing stop calculation.
_LAST_ENTRY_IDX: int = -1


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> object:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    Strategy:
    - Entry: MACD line crosses above Signal line AND close > 50-period SMA
    - Exit: MACD line crosses below Signal line OR close < (highest_since_entry - trailing_mult * ATR)

    Returns either vbt.portfolio.Order(...) for an order or vbt.portfolio.NoOrder for no action.
    """
    global _LAST_ENTRY_IDX

    i = int(c.i)
    pos = float(c.position_now)

    # Safety bounds
    if i < 0:
        return vbt.portfolio.NoOrder

    # Helper checks
    def is_valid(arr: np.ndarray, idx: int) -> bool:
        return 0 <= idx < arr.shape[0] and not np.isnan(arr[idx])

    def cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        if not (is_valid(macd, idx) and is_valid(signal, idx) and is_valid(macd, idx - 1) and is_valid(signal, idx - 1)):
            return False
        return (macd[idx - 1] <= signal[idx - 1]) and (macd[idx] > signal[idx])

    def cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        if not (is_valid(macd, idx) and is_valid(signal, idx) and is_valid(macd, idx - 1) and is_valid(signal, idx - 1)):
            return False
        return (macd[idx - 1] >= signal[idx - 1]) and (macd[idx] < signal[idx])

    # ENTRY: when flat
    if pos == 0.0:
        if cross_up(i) and is_valid(close, i) and is_valid(sma, i):
            if close[i] > sma[i]:
                _LAST_ENTRY_IDX = i
                # Buy using 50% of equity (Percent size_type=2), LongOnly (direction=1)
                return vbt.portfolio.Order(size=float(0.5), size_type=int(2), direction=int(1))
        return vbt.portfolio.NoOrder

    # EXIT: if in position
    # 1) MACD cross down
    if cross_down(i):
        _LAST_ENTRY_IDX = -1
        # Close entire long position: negative infinite size with Percent type
        return vbt.portfolio.Order(size=float('-inf'), size_type=int(2), direction=int(1))

    # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
    entry_idx = _LAST_ENTRY_IDX

    # If not tracked, attempt to find most recent entry signal
    if entry_idx is None or entry_idx < 0 or entry_idx > i:
        found = -1
        for j in range(i, 0, -1):
            if cross_up(j) and is_valid(close, j) and is_valid(sma, j):
                if close[j] > sma[j]:
                    found = j
                    break
        entry_idx = found if found >= 0 else 0

    # Compute highest high since entry
    try:
        highest_since_entry = np.nanmax(high[entry_idx : i + 1])
    except Exception:
        highest_since_entry = np.nan

    if is_valid(atr, i) and not np.isnan(highest_since_entry) and is_valid(close, i):
        thresh = highest_since_entry - (float(trailing_mult) * float(atr[i]))
        if close[i] < thresh:
            _LAST_ENTRY_IDX = -1
            return vbt.portfolio.Order(size=float('-inf'), size_type=int(2), direction=int(1))

    return vbt.portfolio.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators using vectorbt.

    Returns keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_ser = ohlcv['close']
    high_ser = ohlcv['high']
    low_ser = ohlcv['low'] if 'low' in ohlcv.columns else close_ser

    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        'close': close_ser.values,
        'high': high_ser.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }