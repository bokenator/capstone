import numpy as np
import pandas as pd
import vectorbt as vbt


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
    Generate orders based on MACD crossover entries and ATR-based trailing stops.

    This function is intentionally written without numba and works with numpy arrays.
    It relies only on c.i, c.position_now, and c.cash_now from the provided
    OrderContext.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Helper to safely get values (returns np.nan if out of bounds)
    def val(arr: np.ndarray, idx: int):
        if idx < 0 or idx >= arr.shape[0]:
            return np.nan
        return arr[idx]

    # Warmup / not enough data
    if i == 0:
        return (np.nan, 0, 0)

    # Current and previous values
    macd_now = val(macd, i)
    macd_prev = val(macd, i - 1)
    signal_now = val(signal, i)
    signal_prev = val(signal, i - 1)
    close_now = val(close, i)
    sma_now = val(sma, i)
    atr_now = val(atr, i)

    # Helper: check if MACD crossed above at index idx (requires idx >= 1)
    def is_macd_crossed_above_at(idx: int) -> bool:
        if idx <= 0:
            return False
        a = val(macd, idx - 1)
        b = val(signal, idx - 1)
        cval = val(macd, idx)
        d = val(signal, idx)
        if np.isnan(a) or np.isnan(b) or np.isnan(cval) or np.isnan(d):
            return False
        return (a <= b) and (cval > d)

    # Helper: check if MACD crossed below at index idx
    def is_macd_crossed_below_at(idx: int) -> bool:
        if idx <= 0:
            return False
        a = val(macd, idx - 1)
        b = val(signal, idx - 1)
        cval = val(macd, idx)
        d = val(signal, idx)
        if np.isnan(a) or np.isnan(b) or np.isnan(cval) or np.isnan(d):
            return False
        return (a >= b) and (cval < d)

    # ENTRY: if flat
    if pos == 0.0:
        # Need a MACD cross above on this bar and price above SMA
        if is_macd_crossed_above_at(i) and (not np.isnan(sma_now)) and (close_now > sma_now):
            # Enter long with 100% of equity
            return (1.0, 2, 1)
        return (np.nan, 0, 0)

    # EXIT: when in position
    else:
        # 1) MACD crosses below -> exit
        if is_macd_crossed_below_at(i):
            return (-np.inf, 2, 1)

        # 2) Trailing stop based on highest high since entry
        # Find the most recent entry index (where MACD crossed above and price was above SMA)
        entry_idx = None
        for idx in range(i, 0, -1):
            if is_macd_crossed_above_at(idx):
                # ensure price was above SMA at that time
                sma_at_idx = val(sma, idx)
                close_at_idx = val(close, idx)
                if (not np.isnan(sma_at_idx)) and (not np.isnan(close_at_idx)) and (close_at_idx > sma_at_idx):
                    entry_idx = idx
                    break
        # If we couldn't find an entry index, be conservative and do not exit on trailing stop
        if entry_idx is None:
            return (np.nan, 0, 0)

        # Compute highest high since entry (inclusive)
        if entry_idx <= i:
            highs_segment = high[entry_idx:i + 1]
            # use nanmax to ignore NaNs
            try:
                highest_since_entry = np.nanmax(highs_segment)
            except ValueError:
                highest_since_entry = np.nan
        else:
            highest_since_entry = np.nan

        # If ATR missing, cannot compute trailing stop
        if np.isnan(highest_since_entry) or np.isnan(atr_now):
            return (np.nan, 0, 0)

        trailing_level = highest_since_entry - (trailing_mult * atr_now)

        # If price falls below trailing level -> exit
        if close_now < trailing_level:
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy using vectorbt indicator wrappers.

    Returns a dict with numpy arrays: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' and 'close' columns")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']

    # Low may be missing; ATR requires low. Fallback to close if absent.
    if 'low' in ohlcv.columns:
        low_sr = ohlcv['low']
    else:
        low_sr = ohlcv['close']

    # Ensure arrays are float64 and aligned
    close_arr = close_sr.astype('float64').values
    high_arr = high_sr.astype('float64').values
    low_arr = low_sr.astype('float64').values

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values.astype('float64')
    signal_arr = macd_ind.signal.values.astype('float64')

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = atr_ind.atr.values.astype('float64')

    # SMA (moving average)
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = sma_ind.ma.values.astype('float64')

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }
