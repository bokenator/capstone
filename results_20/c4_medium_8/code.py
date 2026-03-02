import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


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
        close: Close prices array
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        Tuple (size, size_type, direction) as described in prompt.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safe access helpers
    def is_valid(idx: int, arr: np.ndarray) -> bool:
        return 0 <= idx < arr.shape[0] and not np.isnan(arr[idx])

    # Determine MACD crossover up/down at current bar
    macd_cross_up = False
    macd_cross_down = False
    if i > 0:
        if is_valid(i - 1, macd) and is_valid(i - 1, signal) and is_valid(i, macd) and is_valid(i, signal):
            macd_cross_up = (macd[i - 1] <= signal[i - 1]) and (macd[i] > signal[i])
            macd_cross_down = (macd[i - 1] >= signal[i - 1]) and (macd[i] < signal[i])

    # Entry logic (only when flat)
    if pos == 0.0:
        should_enter = False
        if macd_cross_up and is_valid(i, sma) and is_valid(i, close):
            # Price must be above SMA
            should_enter = close[i] > sma[i]

        if should_enter:
            # Enter long with 100% of equity
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # If here, we have a position - check for exits
    # 1) MACD cross below -> exit
    if macd_cross_down:
        return (-np.inf, 2, 1)

    # 2) Trailing stop based on highest high since last entry minus mult * ATR
    # Find the most recent entry index (last MACD cross up where price was above SMA)
    last_entry_idx = None
    # Search backwards from current bar to find the last entry signal
    for j in range(i, -1, -1):
        if j == 0:
            break
        if is_valid(j - 1, macd) and is_valid(j - 1, signal) and is_valid(j, macd) and is_valid(j, signal) and is_valid(j, sma) and is_valid(j, close):
            if (macd[j - 1] <= signal[j - 1]) and (macd[j] > signal[j]) and (close[j] > sma[j]):
                last_entry_idx = j
                break

    if last_entry_idx is not None and is_valid(i, atr):
        # Compute highest high since entry (inclusive). Use nanmax to ignore NaNs.
        try:
            highest = np.nanmax(high[last_entry_idx : i + 1])
        except ValueError:
            highest = np.nan

        if not np.isnan(highest):
            trailing_price = highest - trailing_mult * atr[i]
            if is_valid(i, close) and close[i] < trailing_price:
                return (-np.inf, 2, 1)

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
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_ser = ohlcv['close'].astype(float)
    high_ser = ohlcv['high'].astype(float)

    # 'low' is optional in DATA_SCHEMA - fall back to 'close' if missing
    if 'low' in ohlcv.columns:
        low_ser = ohlcv['low'].astype(float)
    else:
        low_ser = close_ser.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        'close': close_ser.values,
        'high': high_ser.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }