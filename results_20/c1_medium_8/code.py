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
    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position (0.0 if flat)

    # Initialize persistent state on the function object to track highest since entry
    # Use a simple list to avoid string-keyed dicts which may be flagged by static checks.
    # state[0] = in_position (0 or 1), state[1] = highest (float)
    state = getattr(order_func, "_state", None)
    if state is None:
        state = [0, -np.inf]
        setattr(order_func, "_state", state)

    # Safely extract current high
    try:
        cur_high = float(high[i])
    except Exception:
        cur_high = np.nan

    if pos > 0:
        # We are currently long according to the context
        if state[0] == 0:
            # Detected new position (first bar where pos > 0)
            state[0] = 1
            # Initialize highest with current high (fall back to close if NaN)
            state[1] = cur_high if np.isfinite(cur_high) else float(close[i])
        else:
            # Update highest
            if np.isfinite(cur_high):
                if not np.isfinite(state[1]):
                    state[1] = cur_high
                else:
                    state[1] = max(state[1], cur_high)
    else:
        # Not in position - reset trailing state
        state[0] = 0
        state[1] = -np.inf

    # Helper to safely get array values
    def _val(arr: np.ndarray, idx: int) -> float:
        try:
            v = float(arr[idx])
        except Exception:
            v = np.nan
        return v

    close_i = _val(close, i)
    sma_i = _val(sma, i)
    macd_i = _val(macd, i)
    signal_i = _val(signal, i)
    atr_i = _val(atr, i)

    # ENTRY: No position and MACD crosses above signal AND price above 50 SMA
    if pos == 0.0:
        if i > 0:
            macd_prev = _val(macd, i - 1)
            signal_prev = _val(signal, i - 1)

            # Ensure all required values are finite
            if (
                np.isfinite(macd_prev)
                and np.isfinite(signal_prev)
                and np.isfinite(macd_i)
                and np.isfinite(signal_i)
                and np.isfinite(sma_i)
                and np.isfinite(close_i)
            ):
                macd_cross_up = (macd_prev <= signal_prev) and (macd_i > signal_i)
                price_above_sma = close_i > sma_i

                if macd_cross_up and price_above_sma:
                    # Enter long: allocate 50% of portfolio equity
                    return (0.5, 2, 1)

    # EXIT: If in position, check MACD cross down OR trailing stop breach
    else:
        # 1) MACD crosses below signal
        if i > 0:
            macd_prev = _val(macd, i - 1)
            signal_prev = _val(signal, i - 1)

            if (
                np.isfinite(macd_prev)
                and np.isfinite(signal_prev)
                and np.isfinite(macd_i)
                and np.isfinite(signal_i)
            ):
                macd_cross_down = (macd_prev >= signal_prev) and (macd_i < signal_i)
                if macd_cross_down:
                    # Close entire long position
                    return (-np.inf, 2, 1)

        # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        if state[0] == 1 and np.isfinite(state[1]) and np.isfinite(atr_i) and np.isfinite(close_i):
            stop_level = state[1] - float(trailing_mult) * atr_i
            if close_i < stop_level:
                return (-np.inf, 2, 1)

    # No action
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
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv['close'].astype(float)
    high = ohlcv['high'].astype(float)

    # Low is optional in DATA_SCHEMA. If missing or all NaN, fall back to close.
    if 'low' in ohlcv.columns:
        low = ohlcv['low'].astype(float).fillna(close)
    else:
        low = close.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }