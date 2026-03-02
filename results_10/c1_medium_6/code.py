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
    trailing_mult: float,
) -> tuple:
    """
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Returns tuple (size, size_type, direction) as required by the backtester wrapper.
    - Buy: (percent_of_equity, 2, 1) where percent_of_equity in (0, 1]
    - Close: (-np.inf, 2, 1) to close entire long position
    - No action: (np.nan, 0, 0)
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safeguard: ensure inputs are numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    # Helper to check for a valid crossover up at index idx
    def is_macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx - 1] <= signal[idx - 1]) and (macd[idx] > signal[idx])

    # Helper to check for a valid crossover down at index idx
    def is_macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx - 1] >= signal[idx - 1]) and (macd[idx] < signal[idx])

    # ENTRY: no position and MACD cross up + price above SMA
    if pos == 0.0:
        # Check indicator availability
        if i > 0 and not np.isnan(close[i]) and not np.isnan(sma[i]) and is_macd_cross_up(i):
            if close[i] > sma[i]:
                # Enter long with 95% of equity
                return (0.95, 2, 1)
        return (np.nan, 0, 0)

    # EXIT: have a long position
    else:
        # 1) Exit on MACD cross down
        if is_macd_cross_down(i):
            return (-np.inf, 2, 1)  # Close entire long position

        # 2) ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        # Find the most recent index <= i where a valid entry signal occurred (MACD cross up and price > SMA)
        try:
            # Build boolean array of entry signals (True at index of entry)
            # We check from index 1 to i inclusive
            start_idx = 1
            end_idx = i
            if end_idx >= start_idx:
                idxs = []
                for idx in range(start_idx, end_idx + 1):
                    if is_macd_cross_up(idx) and (not np.isnan(sma[idx])) and (not np.isnan(close[idx])) and (close[idx] > sma[idx]):
                        idxs.append(idx)
                if len(idxs) > 0:
                    last_entry_idx = idxs[-1]
                else:
                    last_entry_idx = None
            else:
                last_entry_idx = None
        except Exception:
            last_entry_idx = None

        # Compute highest high since entry (inclusive). If no entry index found, use entire history up to i.
        if last_entry_idx is not None:
            highest_since_entry = np.nanmax(high[last_entry_idx : i + 1])
        else:
            highest_since_entry = np.nanmax(high[: i + 1])

        # Ensure ATR at current bar is valid
        if np.isnan(atr[i]) or np.isnan(highest_since_entry) or np.isnan(close[i]):
            return (np.nan, 0, 0)

        trailing_threshold = highest_since_entry - (trailing_mult * atr[i])

        if close[i] < trailing_threshold:
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
    Compute indicators required by the strategy using vectorbt indicator wrappers.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a 1D numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # Use 'low' if available, otherwise fall back to 'close' for ATR calculation
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }
