import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Tuple


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt from_order_func (NO NUMBA).

    Implements MACD crossover entries with ATR-based trailing stops.

    Entry (when flat):
      - MACD crosses above Signal at current bar
      - Close > SMA
      -> Buy with 50% of equity

    Exit (when long):
      - MACD crosses below Signal at current bar -> close
      - Close < (highest_high_since_entry - trailing_mult * ATR) -> close

    Returns tuple (size, size_type, direction) as specified.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safety for first bar
    if i <= 0:
        return (np.nan, 0, 0)

    # Helper: safe access
    def is_valid_num(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and np.isfinite(x)

    # Detect macd crosses at index i (requires i>=1)
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_curr = macd[idx]
        b_curr = signal[idx]
        if not (is_valid_num(a_prev) and is_valid_num(b_prev) and is_valid_num(a_curr) and is_valid_num(b_curr)):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_curr = macd[idx]
        b_curr = signal[idx]
        if not (is_valid_num(a_prev) and is_valid_num(b_prev) and is_valid_num(a_curr) and is_valid_num(b_curr)):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    close_i = close[i]
    sma_i = sma[i]
    atr_i = atr[i] if i < len(atr) else np.nan

    # ENTRY: only when flat
    if pos == 0.0:
        # Check MACD bullish cross and trend filter
        if macd_cross_up(i) and is_valid_num(sma_i) and is_valid_num(close_i) and (close_i > sma_i):
            # Use 50% of equity to enter long
            return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # EXIT: when in a position (>0)
    # 1) MACD bearish cross
    if macd_cross_down(i):
        return (-np.inf, 2, 1)

    # 2) ATR-based trailing stop
    # Find the last bullish MACD cross (with price > SMA) at or before current index.
    # Use it as the entry reference to compute highest high since that bar.
    entry_idx = None
    for k in range(i, 0, -1):
        if macd_cross_up(k):
            # Ensure price > SMA at that time
            if is_valid_num(sma[k]) and is_valid_num(close[k]) and (close[k] > sma[k]):
                entry_idx = k
                break
    # If we didn't find an entry index, fallback to earliest possible (0)
    if entry_idx is None:
        entry_idx = 0

    # Compute highest high since entry (inclusive)
    try:
        # Clip entry_idx to valid range
        entry_idx = max(0, min(entry_idx, len(high) - 1))
        window_highs = high[entry_idx : i + 1]
        # Skip if invalid
        if len(window_highs) == 0:
            return (np.nan, 0, 0)
        # Filter out NaNs when computing max
        valid_highs = window_highs[np.isfinite(window_highs)]
        if valid_highs.size == 0:
            return (np.nan, 0, 0)
        highest_since_entry = float(np.max(valid_highs))
    except Exception:
        return (np.nan, 0, 0)

    # ATR must be valid
    if not is_valid_num(atr_i):
        return (np.nan, 0, 0)

    stop_price = highest_since_entry - (trailing_mult * float(atr_i))

    # If price falls below stop_price -> exit
    if is_valid_num(close_i) and (close_i < stop_price):
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
    Compute indicators needed by the strategy using vectorbt's indicator classes.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are numpy arrays of same length as input.
    """
    # Ensure required columns exist
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    # 'low' may be missing; fallback to 'close' if not present
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    # Extract numpy arrays
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values
    atr_arr = atr_ind.atr.values
    sma_arr = sma_ind.ma.values

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }
