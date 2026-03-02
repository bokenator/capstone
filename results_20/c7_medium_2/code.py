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
    Order function combining MACD crossover entries with ATR-based trailing stops.

    Long-only. Uses 50%% equity on entry (percent sizing = 0.5).

    See prompt for full specification.
    """
    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Helper: safe checks for NaN
    def is_valid(idx, arr):
        try:
            return not np.isnan(arr[idx])
        except Exception:
            return False

    # Helper: MACD cross up at index idx (uses idx-1 and idx)
    def macd_cross_up(idx):
        if idx <= 0:
            return False
        if not (is_valid(idx - 1, macd) and is_valid(idx - 1, signal) and is_valid(idx, macd) and is_valid(idx, signal)):
            return False
        return (macd[idx - 1] <= signal[idx - 1]) and (macd[idx] > signal[idx])

    # Helper: MACD cross down at index idx
    def macd_cross_down(idx):
        if idx <= 0:
            return False
        if not (is_valid(idx - 1, macd) and is_valid(idx - 1, signal) and is_valid(idx, macd) and is_valid(idx, signal)):
            return False
        return (macd[idx - 1] >= signal[idx - 1]) and (macd[idx] < signal[idx])

    # Entry logic (only when flat)
    if pos == 0:
        # Ensure indicators available at current bar
        if macd_cross_up(i) and is_valid(i, close) and is_valid(i, sma) and (close[i] > sma[i]):
            # Enter with 50% of equity
            return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # Exit logic (when in position)
    # 1) MACD cross down at current bar
    if macd_cross_down(i):
        return (-np.inf, 2, 1)  # Close entire long position

    # 2) Trailing stop based on highest price since entry
    # We need to find the most recent entry index j such that the position would have remained open until start of bar i.
    # To avoid relying on external state, we reconstruct entry points using past indicator values only (no lookahead).

    # Build list of candidate entry indices (where MACD crossed up and price > SMA)
    cand_js = []
    # start from 1 because cross requires idx-1
    for j in range(1, i + 1):
        if macd_cross_up(j) and is_valid(j, close) and is_valid(j, sma) and (close[j] > sma[j]):
            cand_js.append(j)

    # Iterate candidates from most recent to oldest
    last_entry_idx = None
    for j in reversed(cand_js):
        # Simulate whether an exit would have occurred strictly before current bar (t in [j+1, i-1])
        exited_before_i = False
        running_max = close[j] if is_valid(j, close) else -np.inf

        # If j == i, then entry happened at this bar (rare while pos>0), treat as not exited before i
        for t in range(j + 1, i):
            # update running max with price at t
            if is_valid(t, close):
                if close[t] > running_max:
                    running_max = close[t]

            # Check MACD bearish cross at t
            if macd_cross_down(t):
                exited_before_i = True
                break

            # Check trailing stop at t (need valid atr)
            if is_valid(t, atr) and running_max != -np.inf and is_valid(t, close):
                if close[t] < (running_max - trailing_mult * atr[t]):
                    exited_before_i = True
                    break

        if not exited_before_i:
            last_entry_idx = j
            break

    if last_entry_idx is None:
        # Fallback: if we cannot determine entry, do not exit on trailing stop
        return (np.nan, 0, 0)

    # Compute highest price since entry up to current bar inclusive
    # Use nanmax to ignore any NaNs
    try:
        highest_since_entry = np.nanmax(close[last_entry_idx:i + 1])
    except Exception:
        highest_since_entry = -np.inf

    # Validate ATR at current bar
    if not is_valid(i, atr):
        return (np.nan, 0, 0)

    # Trailing stop check at current bar
    if is_valid(i, close) and (close[i] < (highest_since_entry - trailing_mult * atr[i])):
        return (-np.inf, 2, 1)

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
    Compute required indicators for the strategy using vectorbt indicators.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input DataFrame index.
    """
    # Validate required columns per DATA_SCHEMA
    if 'close' not in ohlcv.columns:
        raise ValueError("Input ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("Input ohlcv must contain 'high' column")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = atr_ind.atr.values

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }
