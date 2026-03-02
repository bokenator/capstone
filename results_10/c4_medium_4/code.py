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

    Notes:
        - Entry: MACD crosses above Signal AND price > SMA
        - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
        - Uses 50% of equity for entries (size_type=2, size=0.5)
        - Closes whole position using (-np.inf, 2, 1)
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Basic safety checks
    n = len(close)
    if n == 0:
        return (np.nan, 0, 0)

    # Don't act on the very first bar (no previous bar to detect crosses)
    if i < 1:
        return (np.nan, 0, 0)

    # If essential values are NaN, skip
    if (
        np.isnan(macd[i])
        or np.isnan(signal[i])
        or np.isnan(sma[i])
        or np.isnan(close[i])
    ):
        return (np.nan, 0, 0)

    # Helper cross detection for current bar
    def crossed_above(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx < 1:
            return False
        a_prev, b_prev = arr_a[idx - 1], arr_b[idx - 1]
        a_cur, b_cur = arr_a[idx], arr_b[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev <= b_prev) and (a_cur > b_cur)

    def crossed_below(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx < 1:
            return False
        a_prev, b_prev = arr_a[idx - 1], arr_b[idx - 1]
        a_cur, b_cur = arr_a[idx], arr_b[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev >= b_prev) and (a_cur < b_cur)

    # ENTRY: when flat and MACD crosses above Signal AND price > SMA
    if pos == 0.0:
        if crossed_above(macd, signal, i) and (close[i] > sma[i]):
            # Buy with 50% of equity (percent size_type=2)
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # EXIT: when in position
    # 1) MACD crosses below Signal
    if crossed_below(macd, signal, i):
        return (-np.inf, 2, 1)  # Close entire long position

    # 2) Trailing stop based on highest price since entry - trailing_mult * ATR
    # Reconstruct last entry index by scanning past bars for the entry condition.
    last_entry_idx = None
    # We search backwards from current bar to find the most recent bar where
    # MACD crossed above Signal and close > SMA (the entry rule).
    for idx in range(i, 0, -1):
        try:
            if crossed_above(macd, signal, idx) and (not np.isnan(close[idx])) and (not np.isnan(sma[idx])) and (close[idx] > sma[idx]):
                last_entry_idx = idx
                break
        except Exception:
            # In case of unexpected array shapes/values, skip
            continue

    # If we cannot find an entry index, avoid forcing an exit (could be an existing
    # position from before the series start). In that case, do not apply trailing stop.
    if last_entry_idx is None:
        return (np.nan, 0, 0)

    # Ensure ATR is available
    if np.isnan(atr[i]):
        return (np.nan, 0, 0)

    # Compute highest high since entry (inclusive)
    try:
        highest_since_entry = np.nanmax(high[last_entry_idx : i + 1])
    except Exception:
        # Fallback: use current high
        highest_since_entry = high[i]

    if np.isnan(highest_since_entry):
        return (np.nan, 0, 0)

    trailing_stop_price = float(highest_since_entry - trailing_mult * atr[i])

    # If price falls below trailing stop, exit
    if close[i] < trailing_stop_price:
        return (-np.inf, 2, 1)

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
        raise ValueError("OHLCV DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain 'high' column")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']

    # Use 'low' if available, otherwise fallback to 'close' (best-effort)
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(
        close_sr,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # Compute ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    # Extract numpy arrays (1D)
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float).ravel()
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float).ravel()
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float).ravel()
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float).ravel()

    close_arr = np.asarray(close_sr.values, dtype=float).ravel()
    high_arr = np.asarray(high_sr.values, dtype=float).ravel()

    # Ensure all arrays are the same length
    length = len(close_arr)
    if not (len(high_arr) == len(macd_arr) == len(signal_arr) == len(atr_arr) == len(sma_arr) == length):
        raise ValueError("Indicator arrays have mismatching lengths")

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }