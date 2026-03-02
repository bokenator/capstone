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
    Order generation function combining MACD crossover entries with an ATR-based trailing stop.

    Logic:
    - Entry (when flat): MACD crosses above Signal AND price > 50-period SMA
    - Exit (when in position): MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    State management:
    - Uses persistent attributes on the function object to track the highest price since entry
      and the entry index. The state is reset at the start of each run (when c.i == 0).

    Returns a tuple (size, size_type, direction) as specified in the prompt.
    """
    i = int(c.i)
    pos = float(c.position_now)  # 0.0 if flat

    # Initialize or reset state at the start of a run
    if (not hasattr(order_func, "_initialized")) or i == 0:
        order_func._entry_i = None    # index where current position was opened (or planned)
        order_func._highest = np.nan  # highest price seen since entry
        order_func._last_pos = 0.0    # position size from previous call
        order_func._initialized = True

    entry_i = getattr(order_func, "_entry_i", None)
    highest = getattr(order_func, "_highest", np.nan)

    # Safe helpers
    def is_valid_number(x: float) -> bool:
        return not (np.isnan(x) or np.isinf(x))

    # Compute MACD crossover booleans (need previous bar)
    cross_above = False
    cross_below = False
    if i > 0:
        prev_macd = macd[i - 1]
        prev_signal = signal[i - 1]
        curr_macd = macd[i]
        curr_signal = signal[i]

        if is_valid_number(prev_macd) and is_valid_number(prev_signal) and is_valid_number(curr_macd) and is_valid_number(curr_signal):
            prev_diff = prev_macd - prev_signal
            curr_diff = curr_macd - curr_signal
            cross_above = (prev_diff <= 0.0) and (curr_diff > 0.0)
            cross_below = (prev_diff >= 0.0) and (curr_diff < 0.0)

    # If flat -> check entry conditions
    if pos == 0.0:
        # Clear any leftover state when flat
        order_func._entry_i = None
        order_func._highest = np.nan

        # Entry: MACD cross above AND price above SMA
        can_check_entry = is_valid_number(macd[i]) and is_valid_number(signal[i]) and is_valid_number(sma[i]) and is_valid_number(close[i])
        should_enter = False
        if can_check_entry and cross_above and (close[i] > sma[i]):
            should_enter = True

        if should_enter:
            # Use 100% of equity to enter (percent size_type = 2)
            # Record entry state for trailing stop tracking
            order_func._entry_i = i
            order_func._highest = high[i] if is_valid_number(high[i]) else close[i]

            # Update last_pos and return buy order (100% of equity, long only)
            order_func._last_pos = pos
            return (1.0, 2, 1)

        # No entry
        order_func._last_pos = pos
        return (np.nan, 0, 0)

    # If here, we have a position (long)
    # Ensure entry index/ highest are initialized if missing
    if order_func._entry_i is None:
        order_func._entry_i = i
        order_func._highest = high[i] if is_valid_number(high[i]) else close[i]

    # Update running highest since entry
    if not is_valid_number(order_func._highest):
        order_func._highest = high[i] if is_valid_number(high[i]) else close[i]
    else:
        new_high = high[i] if is_valid_number(high[i]) else close[i]
        if is_valid_number(new_high):
            order_func._highest = max(order_func._highest, new_high)

    # Compute trailing stop level if ATR available
    trailing_stop = np.nan
    if is_valid_number(atr[i]) and is_valid_number(order_func._highest):
        trailing_stop = order_func._highest - (trailing_mult * atr[i])

    should_exit_trailing = False
    should_exit_macd = False

    if is_valid_number(trailing_stop) and is_valid_number(close[i]):
        should_exit_trailing = close[i] < trailing_stop

    # MACD cross below check already computed (requires i>0)
    should_exit_macd = cross_below

    if should_exit_trailing or should_exit_macd:
        # Close entire long position
        # Reset state
        order_func._entry_i = None
        order_func._highest = np.nan
        order_func._last_pos = pos
        return (-np.inf, 2, 1)

    # No action
    order_func._last_pos = pos
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
    Compute indicators required by the strategy using vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy ndarray aligned with the input ohlcv index.
    """
    # Validate required columns
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_ser = ohlcv["close"]
    high_ser = ohlcv["high"]
    # Use 'low' if available for ATR, otherwise fallback to 'close' (safe fallback)
    low_ser = ohlcv["low"] if "low" in ohlcv.columns else ohlcv["close"]

    # MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        "close": close_ser.values,
        "high": high_ser.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }
