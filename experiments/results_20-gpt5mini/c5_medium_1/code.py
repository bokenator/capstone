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

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array (use close[c.i] for current price)
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size
        - size_type: int, 0=Amount, 1=Value, 2=Percent
        - direction: int, 0=Both, 1=LongOnly, 2=ShortOnly
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Persistent state stored as a list to avoid accidental column-name detection.
    # state[0] = in_position (bool)
    # state[1] = entry_idx (int or None)
    # state[2] = peak (float)
    if not hasattr(order_func, "_state") or i == 0:
        order_func._state = [False, None, -np.inf]

    state = order_func._state

    def is_valid_index(idx: int) -> bool:
        return 0 <= idx < len(close)

    def safe_get(arr: np.ndarray, idx: int) -> float:
        return float(arr[idx]) if is_valid_index(idx) else np.nan

    def macd_cross_up_at(idx: int) -> bool:
        if idx <= 0:
            return False
        m1 = macd[idx - 1]
        s1 = signal[idx - 1]
        m0 = macd[idx]
        s0 = signal[idx]
        if np.isnan(m1) or np.isnan(s1) or np.isnan(m0) or np.isnan(s0):
            return False
        return (m1 < s1) and (m0 >= s0)

    def macd_cross_down_at(idx: int) -> bool:
        if idx <= 0:
            return False
        m1 = macd[idx - 1]
        s1 = signal[idx - 1]
        m0 = macd[idx]
        s0 = signal[idx]
        if np.isnan(m1) or np.isnan(s1) or np.isnan(m0) or np.isnan(s0):
            return False
        return (m1 > s1) and (m0 <= s0)

    currently_long = pos > 0.0
    prev_in_position = bool(state[0])

    # Detect a new entry in the live portfolio and infer the entry index and peak
    if (not prev_in_position) and currently_long:
        found_idx = None
        for j in range(i, 0, -1):
            if macd_cross_up_at(j):
                cj = close[j]
                sj = sma[j]
                if (not np.isnan(cj)) and (not np.isnan(sj)) and (cj > sj):
                    found_idx = j
                    break
        if found_idx is None:
            found_idx = i

        entry_idx = int(found_idx)
        try:
            peak = float(np.nanmax(high[entry_idx:i + 1]))
            if np.isnan(peak):
                peak = float(high[entry_idx]) if not np.isnan(high[entry_idx]) else -np.inf
        except Exception:
            peak = -np.inf

        state[0] = True
        state[1] = entry_idx
        state[2] = peak

    # Detect a closed position in the live portfolio -> reset
    if prev_in_position and (not currently_long):
        state[0] = False
        state[1] = None
        state[2] = -np.inf

    # Update peak while in position
    if state[0] and currently_long:
        hi = safe_get(high, i)
        if not np.isnan(hi):
            state[2] = max(state[2], hi)

    # Entry check
    should_enter = False
    if pos == 0.0:
        if macd_cross_up_at(i):
            ci = close[i]
            si = sma[i]
            if (not np.isnan(ci)) and (not np.isnan(si)) and (ci > si):
                should_enter = True

    # Exit checks
    should_exit = False
    if pos > 0.0 and macd_cross_down_at(i):
        should_exit = True

    if pos > 0.0 and state[0]:
        peak = state[2]
        atr_i = safe_get(atr, i)
        if (not np.isnan(peak)) and (not np.isneginf(peak)) and (not np.isnan(atr_i)):
            threshold = peak - float(trailing_mult) * float(atr_i)
            if (not np.isnan(close[i])) and (close[i] < threshold):
                should_exit = True

    # Execute orders
    if pos == 0.0 and should_enter:
        return (0.5, 2, 1)

    if pos > 0.0 and should_exit:
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
    Precompute all indicators. Use vectorbt indicator classes.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.
    """
    if "high" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'high' and 'close' columns")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)

    if "low" in ohlcv.columns:
        low = ohlcv["low"].astype(float)
    else:
        low = close.copy()

    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    sma_ind = vbt.MA.run(close, window=sma_period)

    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    return {
        "close": close.values.astype(float),
        "high": high.values.astype(float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }
