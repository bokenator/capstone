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
    trailing_mult: float,
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).
    See prompt for full description of args and return format.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent state on the function object at the start of each run
    # We reset when i == 0 to avoid state leaking across separate backtests/runs.
    if not hasattr(order_func, "_initialized") or i == 0:
        order_func._initialized = True
        order_func._last_pos = 0.0
        order_func._highest = np.nan  # Highest high since entry
        order_func._highest_history = []  # history of highest values per bar

    # Safe access for current values
    # Arrays are expected to be numpy arrays of same length
    close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan
    macd_i = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_i = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    should_enter = False
    should_exit = False

    # Detect MACD crosses using previous bar (no lookahead)
    if i > 0:
        prev_macd = macd[i - 1]
        prev_signal = signal[i - 1]
        # Ensure previous values are valid
        prev_macd_valid = not np.isnan(prev_macd)
        prev_signal_valid = not np.isnan(prev_signal)
    else:
        prev_macd_valid = prev_signal_valid = False

    # ENTRY: Only when flat
    if pos == 0.0:
        if i > 0 and prev_macd_valid and not np.isnan(macd_i) and not np.isnan(signal_i) and not np.isnan(sma_i):
            macd_cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)
            price_above_sma = (not np.isnan(close_i)) and (close_i > sma_i)
            should_enter = bool(macd_cross_up and price_above_sma)

        if should_enter:
            # Enter long using 50% of equity
            # size_type=2 means Percent, direction=1 means LongOnly
            # Update last_pos and history before returning
            order_func._highest_history.append(order_func._highest)
            order_func._last_pos = pos
            return (0.5, 2, 1)

    # POSITION MANAGEMENT when in a position
    else:
        # If we just entered (last_pos == 0), initialize highest to current high
        if order_func._last_pos == 0.0:
            # Initialize highest with current high if available, otherwise use current close
            if not np.isnan(high_i):
                order_func._highest = high_i
            elif not np.isnan(close_i):
                order_func._highest = close_i
            else:
                order_func._highest = np.nan
        else:
            # Update highest if price made a new high
            if not np.isnan(high_i):
                if np.isnan(order_func._highest):
                    order_func._highest = high_i
                else:
                    order_func._highest = max(order_func._highest, high_i)

        # Check for MACD bearish cross
        if i > 0 and prev_macd_valid and not np.isnan(macd_i) and not np.isnan(signal_i):
            macd_cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)
            if macd_cross_down:
                should_exit = True

        # Check trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        if not np.isnan(order_func._highest) and not np.isnan(atr_i) and not np.isnan(close_i):
            trailing_level = order_func._highest - (trailing_mult * atr_i)
            # Only trigger if trailing_level is finite
            if np.isfinite(trailing_level) and close_i < trailing_level:
                should_exit = True

        if should_exit:
            # Close entire long position
            order_func._highest_history.append(order_func._highest)
            order_func._last_pos = pos
            return (-np.inf, 2, 1)

    # No action: update history/state and return no-op
    order_func._highest_history.append(order_func._highest)
    order_func._last_pos = pos

    # If flat, ensure highest is NaN to avoid stale values
    if pos == 0.0:
        order_func._highest = np.nan

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict:
    """
    Precompute all indicators. Use vectorbt indicator classes.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are np.ndarray of same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise KeyError("Input data must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("Input data must contain 'high' column")

    close_series = ohlcv['close'].astype(float)
    high_series = ohlcv['high'].astype(float)

    # Some data may not have 'low' (optional). ATR requires low; fallback to close if missing
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low'].astype(float)
    else:
        # Use close as a conservative fallback
        low_series = close_series.copy()

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA trend filter
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    # Ensure outputs are numpy arrays and have same length
    macd_arr = macd_ind.macd.values.astype(float)
    signal_arr = macd_ind.signal.values.astype(float)
    atr_arr = atr_ind.atr.values.astype(float)
    sma_arr = sma_ind.ma.values.astype(float)
    close_arr = close_series.values.astype(float)
    high_arr = high_series.values.astype(float)

    # Ensure lengths match
    n = len(close_arr)
    assert len(high_arr) == n
    assert len(macd_arr) == n
    assert len(signal_arr) == n
    assert len(atr_arr) == n
    assert len(sma_arr) == n

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }
