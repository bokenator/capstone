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
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent state on the function object.
    # Use attributes instead of dict keys to avoid false positives from static analyzers.
    if (not hasattr(order_func, "_state_initialized")) or i == 0:
        order_func._state_initialized = True
        order_func._in_position = False
        order_func._entry_idx = None
        order_func._highest_since_entry = np.nan

    prev_in_position = bool(getattr(order_func, "_in_position", False))

    # Detect actual position changes reported by the context and update state.
    if (not prev_in_position) and (pos > 0):
        order_func._in_position = True
        order_func._entry_idx = i
        # Initialize highest using today's high (fallback to close if NaN)
        h0 = high[i] if not np.isnan(high[i]) else close[i]
        order_func._highest_since_entry = float(h0) if not np.isnan(h0) else np.nan
        prev_in_position = True

    # If we thought we were in position but the context reports flat, reset state.
    if prev_in_position and pos == 0:
        order_func._in_position = False
        order_func._entry_idx = None
        order_func._highest_since_entry = np.nan
        prev_in_position = False

    # If currently in a position (according to context), update the highest high.
    if pos > 0:
        cur_high = high[i] if not np.isnan(high[i]) else close[i]
        highest = getattr(order_func, "_highest_since_entry", np.nan)
        if np.isnan(highest) and not np.isnan(cur_high):
            order_func._highest_since_entry = float(cur_high)
        elif (not np.isnan(cur_high)):
            try:
                if cur_high > highest:
                    order_func._highest_since_entry = float(cur_high)
            except Exception:
                order_func._highest_since_entry = float(cur_high)

    # Helper functions for detecting MACD crosses
    def macd_crossed_above(idx: int) -> bool:
        if idx == 0:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_cur = macd[idx]
        b_cur = signal[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev <= b_prev) and (a_cur > b_cur)

    def macd_crossed_below(idx: int) -> bool:
        if idx == 0:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_cur = macd[idx]
        b_cur = signal[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev >= b_prev) and (a_cur < b_cur)

    # ENTRY: No position -> check MACD cross above + price above SMA
    if pos == 0:
        try:
            price = close[i]
            sma_val = sma[i]
        except Exception:
            price = np.nan
            sma_val = np.nan

        if macd_crossed_above(i) and (not np.isnan(price)) and (not np.isnan(sma_val)) and (price > sma_val):
            # Enter long using 100% of equity (size_type = Percent)
            return (1.0, 2, 1)

        # No entry
        return (np.nan, 0, 0)

    # POSITION MANAGEMENT: We have a long position -> check exit conditions
    else:
        # Exit condition 1: MACD crosses below signal
        exit_macd = macd_crossed_below(i)

        # Exit condition 2: Price falls below (highest_since_entry - trailing_mult * ATR)
        exit_trail = False
        highest = getattr(order_func, "_highest_since_entry", np.nan)
        atr_val = atr[i] if (i >= 0 and i < len(atr)) else np.nan

        if (not np.isnan(highest)) and (not np.isnan(atr_val)) and (not np.isnan(close[i])):
            stop_level = highest - (trailing_mult * atr_val)
            if close[i] < stop_level:
                exit_trail = True

        if exit_macd or exit_trail:
            # Close entire long position
            return (-np.inf, 2, 1)

        # Otherwise hold
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
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'high' and 'close' columns")

    # Use low if present, otherwise fallback to close (best-effort fallback for ATR)
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(ohlcv['close'], fast_window=macd_fast,
                             slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(ohlcv['high'], low_series, ohlcv['close'], window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(ohlcv['close'], window=sma_period)

    # Extract numpy arrays (preserve length and NaNs)
    close_arr = ohlcv['close'].values.astype(float)
    high_arr = ohlcv['high'].values.astype(float)
    macd_arr = macd_ind.macd.values.astype(float)
    signal_arr = macd_ind.signal.values.astype(float)
    atr_arr = atr_ind.atr.values.astype(float)
    sma_arr = sma_ind.ma.values.astype(float)

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }