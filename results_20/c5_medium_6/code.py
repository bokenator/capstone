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
    pos = float(getattr(c, 'position_now', 0.0))

    # Initialize persistent state on the function object at the start of a run
    # Reset when i == 0 to ensure clean state across backtests
    if not hasattr(order_func, "_last_pos") or i == 0:
        order_func._last_pos = 0.0
        order_func._in_position = False
        order_func._entry_high = -np.inf

    prev_pos = float(getattr(order_func, "_last_pos", 0.0))

    # Update tracking of highest price since entry when in position
    if pos > 0:
        # If we just moved from flat to a position (entry happened on a prior bar),
        # initialize highest_since_entry using current high
        if not getattr(order_func, "_in_position", False):
            order_func._in_position = True
            # Use high if available, otherwise fallback to close
            current_high = float(high[i]) if not np.isnan(high[i]) else float(close[i])
            order_func._entry_high = current_high
        else:
            # Update highest
            if not np.isnan(high[i]) and high[i] > order_func._entry_high:
                order_func._entry_high = float(high[i])
    else:
        # Flat - ensure tracking cleared
        order_func._in_position = False
        order_func._entry_high = -np.inf

    # Store last position for next call
    order_func._last_pos = pos

    # ENTRY: No position and MACD crosses above Signal and price above SMA
    if pos == 0.0:
        # Need at least one prior bar to detect a crossover
        if i > 0:
            # Ensure required values are finite
            if not (np.isnan(macd[i]) or np.isnan(signal[i]) or np.isnan(macd[i-1]) or np.isnan(signal[i-1]) or np.isnan(sma[i]) or np.isnan(close[i])):
                macd_cross_up = (macd[i-1] <= signal[i-1]) and (macd[i] > signal[i])
                price_above_sma = close[i] > sma[i]

                if macd_cross_up and price_above_sma:
                    # Enter long with 50% of equity
                    return (0.5, 2, 1)

    # EXIT: Have position - check MACD cross down or ATR trailing stop breach
    else:
        # MACD bearish cross
        if i > 0:
            if not (np.isnan(macd[i]) or np.isnan(signal[i]) or np.isnan(macd[i-1]) or np.isnan(signal[i-1])):
                macd_cross_down = (macd[i-1] >= signal[i-1]) and (macd[i] < signal[i])
                if macd_cross_down:
                    # Close entire position
                    return (-np.inf, 2, 1)

        # Trailing stop based on highest_since_entry - trailing_mult * ATR
        entry_high = getattr(order_func, "_entry_high", -np.inf)
        # Require a valid entry_high and ATR
        if entry_high != -np.inf and not np.isnan(atr[i]):
            trailing_level = entry_high - float(trailing_mult) * float(atr[i])
            # Trigger exit when price falls below trailing level
            if not np.isnan(close[i]) and close[i] < trailing_level:
                return (-np.inf, 2, 1)

    # No order
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
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    # Use low if provided, otherwise fallback to close to allow ATR computation
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low']
    else:
        low_series = ohlcv['close']

    # Ensure pandas Series inputs (preserve index)
    close_series = pd.Series(close_series).astype(float)
    high_series = pd.Series(high_series).astype(float)
    low_series = pd.Series(low_series).astype(float)

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    return {
        'close': np.asarray(close_series.values, dtype=float),
        'high': np.asarray(high_series.values, dtype=float),
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }