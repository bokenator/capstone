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
    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position size (0.0 if flat)

    # Safety: if index out of bounds for arrays, do nothing
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Helper: get previous values safely
    def prev_val(arr: np.ndarray, idx: int):
        return arr[idx - 1] if idx - 1 >= 0 else np.nan

    macd_prev = prev_val(macd, i)
    signal_prev = prev_val(signal, i)

    macd_curr = macd[i]
    signal_curr = signal[i]

    close_curr = close[i]
    sma_curr = sma[i]
    atr_curr = atr[i]

    # Entry logic: MACD crosses above signal and price > SMA
    should_enter = False
    try:
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and not np.isnan(macd_curr)
            and not np.isnan(signal_curr)
            and macd_prev <= signal_prev
            and macd_curr > signal_curr
            and not np.isnan(sma_curr)
            and close_curr > sma_curr
        ):
            should_enter = True
    except Exception:
        should_enter = False

    # Exit logic: MACD bearish cross
    should_exit_macd = False
    try:
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and not np.isnan(macd_curr)
            and not np.isnan(signal_curr)
            and macd_prev >= signal_prev
            and macd_curr < signal_curr
        ):
            should_exit_macd = True
    except Exception:
        should_exit_macd = False

    # If flat, check for entry
    if pos == 0.0:
        if should_enter:
            # Buy using 50% of equity
            return (0.5, 2, 1)
        else:
            return (np.nan, 0, 0)

    # If in position, check exits
    # 1) MACD bearish cross
    if should_exit_macd:
        return (-np.inf, 2, 1)

    # 2) Trailing stop based on highest price since entry - trailing_mult * ATR
    # Try to get entry index from position record if available using attribute access
    entry_idx = None
    try:
        pr = getattr(c, 'pos_record_now', None)
        if pr is not None:
            # Try common attribute names that may be present on the position record
            for attr_name in ('init_i', 'init_idx', 'init_index', 'initIndex'):
                try:
                    val = getattr(pr, attr_name, None)
                except Exception:
                    val = None
                if val is not None:
                    try:
                        entry_idx = int(val)
                        break
                    except Exception:
                        entry_idx = None
    except Exception:
        entry_idx = None

    # Fallback: if entry_idx not found, attempt to scan backwards to find a reasonable entry point
    if entry_idx is None:
        # Scan backwards up to 500 bars or to start, whichever smaller
        lookback = min(500, n)
        found = False
        for j in range(i, max(i - lookback, 0) - 1, -1):
            if j <= 0:
                break
            m_prev = macd[j - 1]
            s_prev = signal[j - 1]
            m_curr = macd[j]
            s_curr = signal[j]
            if (
                not np.isnan(m_prev)
                and not np.isnan(s_prev)
                and not np.isnan(m_curr)
                and not np.isnan(s_curr)
                and m_prev <= s_prev
                and m_curr > s_curr
            ):
                entry_idx = j
                found = True
                break
        if not found:
            # As ultimate fallback, set entry to 0
            entry_idx = 0

    # Ensure entry_idx is within bounds
    if entry_idx is None:
        entry_idx = 0
    entry_idx = int(max(0, min(entry_idx, i)))

    # Compute highest high since entry (inclusive)
    try:
        if entry_idx <= i:
            highest_since_entry = np.nanmax(high[entry_idx:i + 1])
        else:
            highest_since_entry = high[i]
    except Exception:
        highest_since_entry = high[i]

    # Compute trailing stop level
    trailing_stop = np.nan
    try:
        if not np.isnan(highest_since_entry) and not np.isnan(atr_curr):
            trailing_stop = highest_since_entry - (trailing_mult * atr_curr)
    except Exception:
        trailing_stop = np.nan

    # If price falls below trailing stop, exit
    try:
        if not np.isnan(trailing_stop) and not np.isnan(close_curr) and close_curr < trailing_stop:
            return (-np.inf, 2, 1)
    except Exception:
        pass

    # Otherwise, no action
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
    All values are numpy arrays of same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns or 'low' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high', 'low', and 'close' columns")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']
    low_sr = ohlcv['low']

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