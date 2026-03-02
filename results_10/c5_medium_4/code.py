import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Internal state to track the bar index where the current long position was entered.
# This state is reset at the start of each run (when c.i == 0).
_ENTRY_INDEX: int | None = None


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
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
    global _ENTRY_INDEX

    i = int(c.i)
    pos = float(c.position_now)

    # Reset internal state at the start of each run to avoid cross-run leakage.
    if i == 0:
        _ENTRY_INDEX = None

    # Helper to safely check for valid numeric values
    def is_valid(x: float) -> bool:
        return not (np.isnan(x) or np.isinf(x))

    # ENTRY: Only when flat (no position)
    if pos == 0.0:
        # Need at least one prior bar to detect a MACD crossover
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            cur_macd = macd[i]
            cur_signal = signal[i]

            # Ensure indicator values are valid
            if all(is_valid(val) for val in [prev_macd, prev_signal, cur_macd, cur_signal]):
                macd_cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
            else:
                macd_cross_up = False
        else:
            macd_cross_up = False

        # Trend filter: price above SMA
        cur_sma = sma[i]
        price_above_sma = is_valid(cur_sma) and (close[i] > cur_sma)

        if macd_cross_up and price_above_sma:
            # Place entry order: use 50% of equity to enter long
            # Record entry index immediately so trailing stop starts from this bar
            _ENTRY_INDEX = i
            return (0.5, 2, 1)  # 50% of equity, long only

        return (np.nan, 0, 0)  # No action

    # If we have a position, check exit conditions
    else:
        # Determine entry index for this open position
        entry_idx = _ENTRY_INDEX
        if entry_idx is None:
            # If we didn't record the entry (e.g., started backtest already in position),
            # conservatively assume entry at the beginning
            entry_idx = 0
            _ENTRY_INDEX = entry_idx

        # 1) MACD bearish cross exit
        macd_cross_down = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            cur_macd = macd[i]
            cur_signal = signal[i]

            if all(is_valid(val) for val in [prev_macd, prev_signal, cur_macd, cur_signal]):
                macd_cross_down = (prev_macd >= prev_signal) and (cur_macd < cur_signal)

        if macd_cross_down:
            # Reset entry tracking
            _ENTRY_INDEX = None
            return (-np.inf, 2, 1)  # Close entire long position

        # 2) ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        # Compute highest high since entry up to current bar (inclusive)
        try:
            # Use nan-aware max to handle any NaNs in high
            highest_since_entry = np.nanmax(high[entry_idx : i + 1])
        except Exception:
            highest_since_entry = np.nan

        cur_atr = atr[i]
        if is_valid(highest_since_entry) and is_valid(cur_atr):
            trailing_stop = highest_since_entry - trailing_mult * cur_atr
            # Trigger exit if current close falls below trailing stop
            if close[i] < trailing_stop:
                _ENTRY_INDEX = None
                return (-np.inf, 2, 1)

        # No exit
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

    close = ohlcv['close'].astype(float)
    high = ohlcv['high'].astype(float)

    # Use 'low' if present, otherwise fall back to 'close' (allowed by schema)
    low = ohlcv['low'].astype(float) if 'low' in ohlcv.columns else close

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