import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Optional, Dict, Any

# Module-level state variables to track highest price since entry for trailing stop
_ORDER_IN_POSITION: bool = False
_ORDER_ENTRY_IDX: Optional[int] = None
_ORDER_HIGHEST: float = np.nan


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

    Implements:
      - Entry: MACD crosses above Signal AND price > SMA
      - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Uses module-level state to track the highest high since the last entry.
    """
    global _ORDER_IN_POSITION, _ORDER_ENTRY_IDX, _ORDER_HIGHEST

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the beginning of a run
    if i == 0:
        _ORDER_IN_POSITION = False
        _ORDER_ENTRY_IDX = None
        _ORDER_HIGHEST = np.nan

    # Helper cross detection functions
    def _is_cross_above(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(a[idx]) or np.isnan(b[idx]) or np.isnan(a[idx - 1]) or np.isnan(b[idx - 1]):
            return False
        return (a[idx] > b[idx]) and (a[idx - 1] <= b[idx - 1])

    def _is_cross_below(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(a[idx]) or np.isnan(b[idx]) or np.isnan(a[idx - 1]) or np.isnan(b[idx - 1]):
            return False
        return (a[idx] < b[idx]) and (a[idx - 1] >= b[idx - 1])

    # Keep module-level state consistent with engine's reported position
    if pos == 0.0 and _ORDER_IN_POSITION:
        # Engine flattened the position -> reset
        _ORDER_IN_POSITION = False
        _ORDER_ENTRY_IDX = None
        _ORDER_HIGHEST = np.nan

    if pos > 0.0 and not _ORDER_IN_POSITION:
        # Engine reports an open position but state didn't track it -> initialize
        _ORDER_IN_POSITION = True
        _ORDER_ENTRY_IDX = i
        if not np.isnan(high[i]):
            _ORDER_HIGHEST = float(high[i])
        elif not np.isnan(close[i]):
            _ORDER_HIGHEST = float(close[i])
        else:
            _ORDER_HIGHEST = np.nan

    # Entry logic (when flat)
    if pos == 0.0:
        enter = False
        if _is_cross_above(macd, signal, i):
            if (not np.isnan(close[i])) and (not np.isnan(sma[i])):
                if close[i] > sma[i]:
                    enter = True
        if enter:
            # Set state as if entry will be executed
            _ORDER_IN_POSITION = True
            _ORDER_ENTRY_IDX = i
            if not np.isnan(high[i]):
                _ORDER_HIGHEST = float(high[i])
            elif not np.isnan(close[i]):
                _ORDER_HIGHEST = float(close[i])
            else:
                _ORDER_HIGHEST = np.nan

            # Buy 50% of equity
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # Update highest since entry while in position
    if _ORDER_IN_POSITION:
        cur_high = np.nan
        if not np.isnan(high[i]):
            cur_high = float(high[i])
        elif not np.isnan(close[i]):
            cur_high = float(close[i])

        if np.isnan(_ORDER_HIGHEST):
            _ORDER_HIGHEST = cur_high
        else:
            if not np.isnan(cur_high):
                _ORDER_HIGHEST = float(max(_ORDER_HIGHEST, cur_high))

    # Exit conditions
    # 1) MACD crosses below signal
    if _is_cross_below(macd, signal, i):
        # Reset state
        _ORDER_IN_POSITION = False
        _ORDER_ENTRY_IDX = None
        _ORDER_HIGHEST = np.nan
        return (-np.inf, 2, 1)

    # 2) Trailing stop based on highest since entry
    if not np.isnan(atr[i]) and not np.isnan(_ORDER_HIGHEST):
        trailing_stop = _ORDER_HIGHEST - float(trailing_mult) * float(atr[i])
        if (not np.isnan(close[i])) and (close[i] < trailing_stop):
            _ORDER_IN_POSITION = False
            _ORDER_ENTRY_IDX = None
            _ORDER_HIGHEST = np.nan
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
    Precompute MACD, ATR, and SMA indicators using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input DataFrame.
    """
    # Ensure required columns exist (only those declared in DATA_SCHEMA)
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"Required column '{col}' not found in ohlcv DataFrame")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]
    low_series = ohlcv["low"]

    # MACD (fast, slow, signal)
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (trend filter)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values.astype(float),
        "high": high_series.values.astype(float),
        "macd": macd_ind.macd.values.astype(float),
        "signal": macd_ind.signal.values.astype(float),
        "atr": atr_ind.atr.values.astype(float),
        "sma": sma_ind.ma.values.astype(float),
    }