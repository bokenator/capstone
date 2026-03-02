import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any

# Global state to track trailing stop and entry information per run (keyed by id(close) array)
# State is stored as a list to avoid using string keys that may be misinterpreted by static checks.
# state layout: [in_position (bool), entry_idx (int|None), highest (float)]
_ORDER_STATE: Dict[int, list] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict:
    """
    Compute indicators required by the strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays of the same length as the input DataFrame.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # Handle optional 'low' column - required for ATR. If missing, fallback to close.
    if 'low' in ohlcv.columns:
        low = ohlcv['low']
    else:
        # Fallback (keeps computations working even if low is absent)
        low = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    # Extract numpy arrays
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values
    atr_arr = atr_ind.atr.values
    sma_arr = sma_ind.ma.values

    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }


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
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Logic:
    - Entry (when flat): MACD cross above Signal AND close > SMA
    - Exit (when in position): MACD cross below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction) as required by the backtest wrapper.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Key the state by the identity of the price array (unique per run)
    key = id(close)

    # Initialize state at start of a run (or if missing)
    if key not in _ORDER_STATE or i == 0:
        _ORDER_STATE[key] = [False, None, -np.inf]

    state = _ORDER_STATE[key]

    # Helper to safely read arrays
    def _safe_get(arr: np.ndarray, idx: int):
        try:
            return arr[idx]
        except Exception:
            return np.nan

    close_i = float(_safe_get(close, i))
    high_i = float(_safe_get(high, i))
    macd_i = _safe_get(macd, i)
    signal_i = _safe_get(signal, i)
    atr_i = _safe_get(atr, i)
    sma_i = _safe_get(sma, i)

    # If currently long, update highest_since_entry and check exit conditions
    if pos > 0:
        # state[0] == in_position flag
        if not state[0]:
            state[0] = True
            state[1] = i
            state[2] = high_i if not np.isnan(high_i) else -np.inf

        # Update highest price seen since entry
        if not np.isnan(high_i):
            state[2] = max(state[2], float(high_i))

        # 1) MACD bearish cross exit
        macd_bear_cross = False
        if i > 0:
            prev_macd = _safe_get(macd, i - 1)
            prev_signal = _safe_get(signal, i - 1)
            if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(macd_i) or np.isnan(signal_i)):
                if prev_macd >= prev_signal and macd_i < signal_i:
                    macd_bear_cross = True

        # 2) Trailing stop exit
        trail_price = np.nan
        trail_exit = False
        if not np.isnan(state[2]) and not np.isnan(atr_i):
            trail_price = state[2] - trailing_mult * float(atr_i)
            if not np.isnan(close_i) and close_i < trail_price:
                trail_exit = True

        if macd_bear_cross or trail_exit:
            # Reset state on exit
            _ORDER_STATE[key] = [False, None, -np.inf]
            return (-np.inf, 2, 1)  # Close entire long position (percent)

        # No action (hold)
        return (np.nan, 0, 0)

    # No position: check entry conditions
    else:
        macd_bull_cross = False
        if i > 0:
            prev_macd = _safe_get(macd, i - 1)
            prev_signal = _safe_get(signal, i - 1)
            if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(macd_i) or np.isnan(signal_i)):
                if prev_macd <= prev_signal and macd_i > signal_i:
                    macd_bull_cross = True

        above_sma = False
        if not np.isnan(close_i) and not np.isnan(sma_i):
            if close_i > sma_i:
                above_sma = True

        if macd_bull_cross and above_sma:
            # Enter full allocation (100% of equity)
            _ORDER_STATE[key] = [True, i, high_i if not np.isnan(high_i) else -np.inf]
            return (1.0, 2, 1)

        return (np.nan, 0, 0)
