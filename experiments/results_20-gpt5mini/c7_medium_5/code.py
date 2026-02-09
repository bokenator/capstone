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

    # Initialize per-run state on the first call
    # We store small amount of state on the function object itself but reset at each run start (i == 0)
    if i == 0:
        # previous position observed (to detect new entries/exits)
        setattr(order_func, "_prev_pos", float(pos))
        # highest price since the last entry
        setattr(order_func, "_highest_since_entry", np.nan)

    prev_pos = float(getattr(order_func, "_prev_pos", 0.0))

    # Helpers to safely get current and previous indicator values
    def _val(arr: np.ndarray, idx: int):
        try:
            return arr[idx]
        except Exception:
            return np.nan

    cur_macd = float(_val(macd, i))
    cur_signal = float(_val(signal, i))
    prev_macd = float(_val(macd, i - 1)) if i > 0 else np.nan
    prev_signal = float(_val(signal, i - 1)) if i > 0 else np.nan

    cur_close = float(_val(close, i))
    cur_high = float(_val(high, i))
    cur_atr = float(_val(atr, i))
    cur_sma = float(_val(sma, i))

    # If currently in position -> update highest_since_entry and check exit conditions
    if pos > 0.0:
        # Detect newly opened position (prev_pos == 0 but pos > 0)
        if prev_pos == 0.0:
            # Set initial highest to current high (use close as fallback)
            initial_high = cur_high if not np.isnan(cur_high) else cur_close
            setattr(order_func, "_highest_since_entry", float(initial_high))
        else:
            # Update highest if price makes new highs
            highest = float(getattr(order_func, "_highest_since_entry", np.nan))
            if np.isnan(highest):
                highest = cur_high if not np.isnan(cur_high) else cur_close
            else:
                if not np.isnan(cur_high):
                    highest = max(highest, cur_high)
            setattr(order_func, "_highest_since_entry", float(highest))

        # 1) Exit on MACD bearish cross: previous MACD >= previous Signal AND current MACD < current Signal
        if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(cur_macd) or np.isnan(cur_signal)):
            if (prev_macd >= prev_signal) and (cur_macd < cur_signal):
                # Close entire long position
                setattr(order_func, "_prev_pos", pos)
                return (-np.inf, 2, 1)

        # 2) Exit on ATR-based trailing stop
        highest = float(getattr(order_func, "_highest_since_entry", np.nan))
        if not np.isnan(highest) and not np.isnan(cur_atr):
            stop_level = highest - float(trailing_mult) * cur_atr
            # Exit when price falls strictly below stop level
            if cur_close < stop_level:
                setattr(order_func, "_prev_pos", pos)
                return (-np.inf, 2, 1)

    else:
        # No position - check entry conditions
        # Entry requires MACD crossing above signal AND price above SMA
        if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(cur_macd) or np.isnan(cur_signal) or np.isnan(cur_sma)):
            macd_cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
            price_above_sma = cur_close > cur_sma
            if macd_cross_up and price_above_sma:
                # Enter using 50% of equity (percent size type)
                setattr(order_func, "_prev_pos", pos)
                return (0.5, 2, 1)

    # Update prev_pos for next call
    setattr(order_func, "_prev_pos", pos)
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
        raise ValueError("`ohlcv` must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("`ohlcv` must contain 'high' column")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']
    # Use low column if present; otherwise fall back to close (safeguard)
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = np.asarray(macd_ind.macd)
    signal_arr = np.asarray(macd_ind.signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = np.asarray(atr_ind.atr)

    # Compute SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = np.asarray(sma_ind.ma)

    # Prepare outputs and forward-fill initial NaNs (no lookahead)
    def _ffill(arr: np.ndarray) -> np.ndarray:
        s = pd.Series(arr)
        s = s.fillna(method='ffill')
        return s.to_numpy()

    macd_arr = _ffill(macd_arr)
    signal_arr = _ffill(signal_arr)
    atr_arr = _ffill(atr_arr)
    sma_arr = _ffill(sma_arr)

    return {
        'close': np.asarray(close_sr.to_numpy(), dtype=float),
        'high': np.asarray(high_sr.to_numpy(), dtype=float),
        'macd': np.asarray(macd_arr, dtype=float),
        'signal': np.asarray(signal_arr, dtype=float),
        'atr': np.asarray(atr_arr, dtype=float),
        'sma': np.asarray(sma_arr, dtype=float),
    }
