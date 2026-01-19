import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


class _EntryState:
    """Simple container to avoid using string-keyed dicts which may be
    confused with DataFrame column access by static analyzers.
    """

    def __init__(self) -> None:
        self.entry_index: Any = None
        self.highest: Any = None


# Module-level state to track entry/highest price since entry across order_func calls.
_ENTRY_STATE = _EntryState()


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
    See prompt for detailed description of parameters and return format.
    """
    global _ENTRY_STATE

    i = int(c.i)
    pos = float(c.position_now)

    # Reset internal state at the start of a backtest/run
    if i == 0 and pos == 0:
        _ENTRY_STATE.entry_index = None
        _ENTRY_STATE.highest = None

    # Safety checks for array bounds and NaNs
    def _is_finite(val: float) -> bool:
        return np.isfinite(val)

    def _cross_up(idx: int) -> bool:
        if idx == 0:
            return False
        prev_m, prev_s = macd[idx - 1], signal[idx - 1]
        cur_m, cur_s = macd[idx], signal[idx]
        if not (_is_finite(prev_m) and _is_finite(prev_s) and _is_finite(cur_m) and _is_finite(cur_s)):
            return False
        return (prev_m <= prev_s) and (cur_m > cur_s)

    def _cross_down(idx: int) -> bool:
        if idx == 0:
            return False
        prev_m, prev_s = macd[idx - 1], signal[idx - 1]
        cur_m, cur_s = macd[idx], signal[idx]
        if not (_is_finite(prev_m) and _is_finite(prev_s) and _is_finite(cur_m) and _is_finite(cur_s)):
            return False
        return (prev_m >= prev_s) and (cur_m < cur_s)

    # No position: check entry conditions
    if pos == 0.0:
        # Ensure required indicator values exist for this bar
        if not (_is_finite(close[i]) and _is_finite(high[i]) and _is_finite(sma[i])):
            return vbt.portfolio.enums.NoOrder

        should_enter = _cross_up(i) and (close[i] > sma[i])

        if should_enter:
            # Initialize tracking for trailing stop
            _ENTRY_STATE.entry_index = i
            _ENTRY_STATE.highest = float(high[i]) if _is_finite(high[i]) else float(close[i])
            # Enter using 50% of equity (Percent size_type = 2), Long only (direction = 1)
            # Create an order using vectorbt's order_nb
            return vbt.portfolio.nb.order_nb(size=0.5, price=np.inf, size_type=2, direction=1)

        return vbt.portfolio.enums.NoOrder

    # Have a long position: check exit conditions
    else:
        # Ensure arrays have valid values
        if not _is_finite(high[i]):
            # If high is missing/NaN, can't update trailing high; skip action
            return vbt.portfolio.enums.NoOrder

        # Initialize highest if missing (robustness for unexpected state)
        if _ENTRY_STATE.highest is None:
            _ENTRY_STATE.highest = float(high[i])
        else:
            # Update highest since entry
            _ENTRY_STATE.highest = float(np.maximum(_ENTRY_STATE.highest, high[i]))

        # Trailing stop threshold calculation
        atr_val = atr[i]
        highest = _ENTRY_STATE.highest

        trailing_hit = False
        if (_is_finite(highest) and _is_finite(atr_val) and _is_finite(close[i])):
            stop_level = highest - (trailing_mult * atr_val)
            trailing_hit = close[i] < stop_level

        macd_cross_down = _cross_down(i)

        if macd_cross_down or trailing_hit:
            # Clear state because we will close the position
            _ENTRY_STATE.entry_index = None
            _ENTRY_STATE.highest = None
            # Close entire long position using percentage (-inf means full close)
            return vbt.portfolio.nb.order_nb(size=-np.inf, price=np.inf, size_type=2, direction=1)

        return vbt.portfolio.enums.NoOrder


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

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are numpy arrays of the same length as the input.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_ser = ohlcv["close"]
    high_ser = ohlcv["high"]
    low_ser = ohlcv["low"] if "low" in ohlcv.columns else close_ser

    # MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        "close": close_ser.values,
        "high": high_ser.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }
