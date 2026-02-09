import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple


def order_func(
    c: Any,
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

    Behavior:
        - Entry: when MACD crosses above signal and close > sma -> buy with 50% equity
        - Exit: when MACD crosses below signal OR close < (highest_since_entry - trailing_mult * ATR)

    Notes:
        - Maintains minimal Python-level state across calls to track the highest price since
          an actual position entry (required for the ATR-based trailing stop).
        - State is reset at the start of a run (when c.i == 0).
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize or reset persistent attributes on the first bar of a run
    if (not hasattr(order_func, "_initialized")) or i == 0:
        order_func._prev_pos = 0.0
        order_func._entry_idx = None
        order_func._highest = None
        order_func._initialized = True

    prev_pos = float(getattr(order_func, "_prev_pos", 0.0))
    highest = getattr(order_func, "_highest", None)

    # Detect actual entry: transition from flat to long
    if prev_pos == 0.0 and pos > 0.0:
        order_func._entry_idx = i
        h_val = high[i] if (not np.isnan(high[i])) else close[i]
        order_func._highest = float(h_val) if not np.isnan(h_val) else None
        highest = order_func._highest

    # If already in position, update the highest high since entry
    if pos > 0.0:
        if highest is None:
            h_val = high[i] if (not np.isnan(high[i])) else close[i]
            order_func._highest = float(h_val) if not np.isnan(h_val) else None
            highest = order_func._highest
        else:
            if not np.isnan(high[i]):
                order_func._highest = max(highest, float(high[i]))
                highest = order_func._highest

    # If we detect an exit (prev_pos > 0 and pos == 0), clear entry-specific attributes
    if prev_pos > 0.0 and pos == 0.0:
        order_func._entry_idx = None
        order_func._highest = None
        highest = None

    # ENTRY LOGIC (only when flat)
    if pos == 0.0:
        should_enter = False
        if i > 0:
            if (
                not np.isnan(macd[i])
                and not np.isnan(signal[i])
                and not np.isnan(macd[i - 1])
                and not np.isnan(signal[i - 1])
                and not np.isnan(sma[i])
                and not np.isnan(close[i])
            ):
                macd_cross_up = (macd[i - 1] <= signal[i - 1]) and (macd[i] > signal[i])
                price_above_sma = close[i] > sma[i]
                should_enter = bool(macd_cross_up and price_above_sma)

        if should_enter:
            order_func._prev_pos = pos
            return (0.5, 2, 1)

    # EXIT LOGIC (when in a long position)
    else:
        should_exit = False

        # MACD cross below
        if i > 0:
            if (
                not np.isnan(macd[i])
                and not np.isnan(signal[i])
                and not np.isnan(macd[i - 1])
                and not np.isnan(signal[i - 1])
            ):
                macd_cross_down = (macd[i - 1] >= signal[i - 1]) and (macd[i] < signal[i])
                if macd_cross_down:
                    should_exit = True

        # ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        if (not should_exit) and (highest is not None):
            if (not np.isnan(atr[i])) and (not np.isnan(close[i])) and (highest is not None):
                stop_price = highest - (trailing_mult * atr[i])
                if close[i] < stop_price:
                    should_exit = True

        if should_exit:
            order_func._prev_pos = pos
            return (-np.inf, 2, 1)

    # No action
    order_func._prev_pos = pos
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
    # Validate required columns from DATA_SCHEMA
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # Provide fallback for 'low' if missing (use close as fallback)
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    # Extract numpy arrays (ensure float dtype)
    macd_arr = macd_ind.macd.values.astype(float)
    signal_arr = macd_ind.signal.values.astype(float)
    atr_arr = atr_ind.atr.values.astype(float)
    sma_arr = sma_ind.ma.values.astype(float)

    return {
        'close': close.values.astype(float),
        'high': high.values.astype(float),
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }