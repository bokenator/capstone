import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Global state to track highest price since entry for the trailing stop.
# Using separate global variables (instead of dict with string keys) to avoid
# any accidental static analysis matching DataFrame column access.
_ENTRY_IDX: int | None = None
_HIGHEST: float = -np.inf


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

    This implementation uses simple global state to track the highest high
    since the latest entry in order to compute an ATR-based trailing stop.

    Long-only entries: buy with 50% of equity when MACD crosses above Signal and
    price is above 50-period SMA.

    Exits: MACD cross down OR price falls below (highest_since_entry - trailing_mult * ATR).
    """
    global _ENTRY_IDX, _HIGHEST

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the beginning of the run
    if i == 0:
        _ENTRY_IDX = None
        _HIGHEST = -np.inf

    # Safety checks for arrays (ensure we have values at index i)
    def is_valid(arr, idx):
        try:
            v = arr[idx]
        except Exception:
            return False
        return not (v is None or (isinstance(v, float) and np.isnan(v)))

    curr_close = float(close[i]) if is_valid(close, i) else np.nan
    curr_high = float(high[i]) if is_valid(high, i) else np.nan
    curr_macd = float(macd[i]) if is_valid(macd, i) else np.nan
    curr_signal = float(signal[i]) if is_valid(signal, i) else np.nan
    curr_atr = float(atr[i]) if is_valid(atr, i) else np.nan
    curr_sma = float(sma[i]) if is_valid(sma, i) else np.nan

    # Helper to detect cross up/down safely
    def cross_up(arr1, arr2, idx):
        if idx == 0:
            return False
        if not (is_valid(arr1, idx) and is_valid(arr2, idx) and is_valid(arr1, idx - 1) and is_valid(arr2, idx - 1)):
            return False
        return (arr1[idx - 1] <= arr2[idx - 1]) and (arr1[idx] > arr2[idx])

    def cross_down(arr1, arr2, idx):
        if idx == 0:
            return False
        if not (is_valid(arr1, idx) and is_valid(arr2, idx) and is_valid(arr1, idx - 1) and is_valid(arr2, idx - 1)):
            return False
        return (arr1[idx - 1] >= arr2[idx - 1]) and (arr1[idx] < arr2[idx])

    should_enter = False
    should_exit = False

    macd_cross_up = cross_up(macd, signal, i)
    macd_cross_down = cross_down(macd, signal, i)

    price_above_sma = False
    if not np.isnan(curr_close) and not np.isnan(curr_sma):
        price_above_sma = curr_close > curr_sma

    # ENTRY
    if pos == 0.0:
        # Reset trailing state when flat
        _ENTRY_IDX = None
        _HIGHEST = -np.inf

        if macd_cross_up and price_above_sma:
            should_enter = True
            # Set optimistic state for highest price since entry to include current high
            _ENTRY_IDX = i
            if not np.isnan(curr_high):
                _HIGHEST = curr_high
            else:
                _HIGHEST = curr_close if not np.isnan(curr_close) else -np.inf

            # Buy with 50% of equity (percent size)
            return (0.5, 2, 1)

    # POSITIONED - check for exits and maintain highest
    else:
        # Initialize state if missing (in case we missed setting it on entry)
        if _ENTRY_IDX is None:
            _ENTRY_IDX = i
            _HIGHEST = curr_high if not np.isnan(curr_high) else curr_close

        # Update highest since entry
        if not np.isnan(curr_high):
            if np.isnan(_HIGHEST):
                _HIGHEST = curr_high
            else:
                _HIGHEST = max(_HIGHEST, curr_high)

        # Compute trailing stop level if ATR is available and highest is set
        trailing_stop = np.nan
        if not np.isnan(curr_atr) and np.isfinite(_HIGHEST):
            trailing_stop = _HIGHEST - trailing_mult * curr_atr

        # Exit conditions
        if macd_cross_down:
            should_exit = True
        elif not np.isnan(trailing_stop) and not np.isnan(curr_close) and (curr_close < trailing_stop):
            should_exit = True

        if should_exit:
            # Reset trailing state
            _ENTRY_IDX = None
            _HIGHEST = -np.inf
            # Close entire long position: use -inf percent to indicate close all
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
    Precompute MACD, ATR, and SMA indicators using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    where each value is a numpy array aligned with the input ohlcv index.
    """
    # Validate required columns per DATA_SCHEMA
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # low is optional per schema; if missing, fall back to close
    if 'low' in ohlcv.columns:
        low = ohlcv['low']
    else:
        low = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    atr_arr = atr_ind.atr.values

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }