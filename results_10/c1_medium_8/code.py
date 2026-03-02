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
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent (of equity)
        - direction: int, 0=Both, 1=LongOnly, 2=ShortOnly
    """
    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position (0.0 if flat)

    # Use function attributes to persist simple state between calls.
    # Avoid using arbitrary string keys to satisfy static schema checks.
    if not hasattr(order_func, "_entry_idx"):
        order_func._entry_idx = None
    if not hasattr(order_func, "_last_i"):
        order_func._last_i = -1

    entry_idx = order_func._entry_idx
    last_i = order_func._last_i

    # Reset state if a new run or if index went backwards
    if i == 0 or i <= last_i:
        entry_idx = None
    order_func._last_i = i

    # Helper to safely get value or np.nan if out of bounds
    def safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return np.nan

    # Ensure inputs are numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    # Clear stored entry index if no position exists
    if pos == 0.0 and entry_idx is not None:
        entry_idx = None

    # Entry logic (only when flat)
    if pos == 0.0:
        # Need at least one previous bar for crossover detection
        if i > 0:
            prev_macd = safe_get(macd, i - 1)
            prev_signal = safe_get(signal, i - 1)
            cur_macd = safe_get(macd, i)
            cur_signal = safe_get(signal, i)
            cur_close = safe_get(close, i)
            cur_sma = safe_get(sma, i)

            # Check for valid numbers
            if not any(np.isnan([prev_macd, prev_signal, cur_macd, cur_signal, cur_close, cur_sma])):
                macd_cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
                price_above_sma = cur_close > cur_sma

                if macd_cross_up and price_above_sma:
                    # Place a long entry using 100% of equity
                    # Record tentative entry index. It will be cleared if the position is not opened
                    entry_idx = i
                    order_func._entry_idx = entry_idx
                    return (1.0, 2, 1)

    else:
        # We have a position - check for exit conditions
        # 1) MACD cross below signal
        macd_cross_down = False
        if i > 0:
            prev_macd = safe_get(macd, i - 1)
            prev_signal = safe_get(signal, i - 1)
            cur_macd = safe_get(macd, i)
            cur_signal = safe_get(signal, i)

            if not any(np.isnan([prev_macd, prev_signal, cur_macd, cur_signal])):
                macd_cross_down = (prev_macd >= prev_signal) and (cur_macd < cur_signal)

            if macd_cross_down:
                # Close entire long position
                entry_idx = None
                order_func._entry_idx = entry_idx
                return (-np.inf, 2, 1)

        # 2) ATR-based trailing stop using highest high since entry
        if entry_idx is None:
            # Find the most recent bar where entry conditions occurred and there was no intervening MACD exit
            inferred = None
            # Walk backwards to find a potential entry
            for j in range(i, 0, -1):
                prev_m = safe_get(macd, j - 1)
                prev_s = safe_get(signal, j - 1)
                cur_m = safe_get(macd, j)
                cur_s = safe_get(signal, j)
                price_j = safe_get(close, j)
                sma_j = safe_get(sma, j)

                if any(np.isnan([prev_m, prev_s, cur_m, cur_s, price_j, sma_j])):
                    continue

                if (prev_m <= prev_s) and (cur_m > cur_s) and (price_j > sma_j):
                    # Check for intervening MACD exit between j+1 and i
                    had_exit = False
                    for k in range(j + 1, i + 1):
                        if k == 0:
                            continue
                        pm = safe_get(macd, k - 1)
                        ps = safe_get(signal, k - 1)
                        cm = safe_get(macd, k)
                        cs = safe_get(signal, k)
                        if any(np.isnan([pm, ps, cm, cs])):
                            continue
                        if (pm >= ps) and (cm < cs):
                            had_exit = True
                            break
                    if not had_exit:
                        inferred = j
                        break
            entry_idx = inferred
            order_func._entry_idx = entry_idx

        # If we have an entry index and ATR available, compute trailing stop
        if entry_idx is not None:
            start = int(entry_idx)
            if start < 0:
                start = 0
            stop = i + 1
            # Slice highs from entry to current bar (inclusive)
            highs_since_entry = high[start:stop]
            # Compute highest high safely
            if highs_since_entry.size == 0:
                highest = safe_get(close, i)
            else:
                highest = np.nanmax(highs_since_entry)
                if np.isnan(highest):
                    highest = np.nanmax(close[start:stop])
                    if np.isnan(highest):
                        highest = safe_get(close, i)

            atr_now = safe_get(atr, i)
            if not np.isnan(atr_now) and not np.isnan(highest):
                trailing_price = highest - float(trailing_mult) * float(atr_now)
                cur_close = safe_get(close, i)
                if not np.isnan(cur_close) and cur_close < trailing_price:
                    entry_idx = None
                    order_func._entry_idx = entry_idx
                    return (-np.inf, 2, 1)

    # Persist any changes to entry_idx
    order_func._entry_idx = entry_idx

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
    # Validate required columns
    if 'high' not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain 'high' column")
    if 'close' not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain 'close' column")

    high_sr = ohlcv['high']
    close_sr = ohlcv['close']

    # low may be missing in DATA_SCHEMA; fall back to close if absent
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else close_sr

    # Compute MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }