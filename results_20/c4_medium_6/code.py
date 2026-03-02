import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


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
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Safety checks for index bounds
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper to check for valid numeric value
    def is_valid(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    # Entry logic: MACD crosses above signal and price above SMA
    def is_macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        if not all(is_valid(v) for v in (macd[idx], signal[idx], macd[idx - 1], signal[idx - 1])):
            return False
        return (macd[idx] > signal[idx]) and (macd[idx - 1] <= signal[idx - 1])

    def is_macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        if not all(is_valid(v) for v in (macd[idx], signal[idx], macd[idx - 1], signal[idx - 1])):
            return False
        return (macd[idx] < signal[idx]) and (macd[idx - 1] >= signal[idx - 1])

    # If flat, check for entry
    if pos == 0.0:
        if is_macd_cross_up(i) and is_valid(sma[i]) and is_valid(close[i]):
            if close[i] > sma[i]:
                # Enter: use 50% of equity (percent size type)
                return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # If long, check for exit conditions
    # 1) MACD crosses below signal
    if pos > 0.0:
        if is_macd_cross_down(i):
            # Close entire long position
            return (-np.inf, 2, 1)

        # 2) ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        # Find the most recent entry index where MACD crossed up and price was above SMA
        entry_idx = None
        for k in range(i, 0, -1):
            if is_macd_cross_up(k) and is_valid(sma[k]) and is_valid(close[k]) and (close[k] > sma[k]):
                entry_idx = k
                break

        if entry_idx is None:
            # Could not determine entry index; skip trailing stop evaluation
            return (np.nan, 0, 0)

        # Compute highest high since entry (inclusive)
        try:
            segment_highs = high[entry_idx : i + 1]
            if len(segment_highs) == 0:
                return (np.nan, 0, 0)
            highest_since_entry = np.nanmax(segment_highs)
        except Exception:
            return (np.nan, 0, 0)

        # Ensure ATR is available at current bar
        if not is_valid(atr[i]) or not is_valid(highest_since_entry):
            return (np.nan, 0, 0)

        trailing_stop = highest_since_entry - float(trailing_mult) * float(atr[i])

        # If price falls below trailing stop, exit
        if is_valid(close[i]) and (close[i] < trailing_stop):
            return (-np.inf, 2, 1)

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
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # Use 'low' if provided, otherwise fall back to 'close' to allow ATR calculation
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else close_series

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_arr = atr_ind.atr.values

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }