import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict

# Module-level state variables (avoid using arbitrary string literals to satisfy schema checks)
_IN_POSITION: bool = False
_ENTRY_INDEX: int | None = None
_HIGHEST_SINCE_ENTRY: float = -np.inf


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

    This pure Python (non-numba) function maintains simple module-level state
    to track the highest price since entry for the ATR-based trailing stop.
    """
    global _IN_POSITION, _ENTRY_INDEX, _HIGHEST_SINCE_ENTRY

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the beginning of a run
    if i == 0:
        _IN_POSITION = False
        _ENTRY_INDEX = None
        _HIGHEST_SINCE_ENTRY = -np.inf

    # Defensive bounds checks for array accesses
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    def safe_val(arr: np.ndarray, idx: int) -> float:
        try:
            v = arr[idx]
        except Exception:
            return np.nan
        return float(v) if not pd.isna(v) else np.nan

    curr_close = safe_val(close, i)
    curr_high = safe_val(high, i) if high is not None else curr_close
    curr_atr = safe_val(atr, i)
    curr_sma = safe_val(sma, i)

    # If a position exists in the execution context but our module state doesn't know
    # about it (e.g., backtest started with an open position), initialize state.
    if pos > 0 and not _IN_POSITION:
        _IN_POSITION = True
        _ENTRY_INDEX = 0
        try:
            _HIGHEST_SINCE_ENTRY = float(np.nanmax(high[: i + 1]))
        except Exception:
            _HIGHEST_SINCE_ENTRY = curr_high if not np.isnan(curr_high) else curr_close

    # If we are flat, ensure state reflects that
    if pos == 0:
        _IN_POSITION = False
        _ENTRY_INDEX = None
        _HIGHEST_SINCE_ENTRY = -np.inf

    # Update highest while in position
    if _IN_POSITION:
        if np.isnan(_HIGHEST_SINCE_ENTRY):
            _HIGHEST_SINCE_ENTRY = curr_high if not np.isnan(curr_high) else curr_close
        else:
            if not np.isnan(curr_high):
                _HIGHEST_SINCE_ENTRY = max(_HIGHEST_SINCE_ENTRY, curr_high)

    # Helper functions for MACD crosses
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a0 = safe_val(macd, idx - 1)
        b0 = safe_val(signal, idx - 1)
        a1 = safe_val(macd, idx)
        b1 = safe_val(signal, idx)
        if any(pd.isna(x) for x in (a0, b0, a1, b1)):
            return False
        return (a0 <= b0) and (a1 > b1)

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a0 = safe_val(macd, idx - 1)
        b0 = safe_val(signal, idx - 1)
        a1 = safe_val(macd, idx)
        b1 = safe_val(signal, idx)
        if any(pd.isna(x) for x in (a0, b0, a1, b1)):
            return False
        return (a0 >= b0) and (a1 < b1)

    # Trailing stop check
    trailing_hit = False
    if _IN_POSITION and not np.isnan(curr_atr):
        highest = _HIGHEST_SINCE_ENTRY
        if not np.isnan(highest) and highest != -np.inf:
            stop_level = highest - (trailing_mult * curr_atr)
            if (not np.isnan(curr_close)) and (curr_close < stop_level):
                trailing_hit = True

    # ENTRY: flat and MACD cross up and price above SMA
    if pos == 0:
        if macd_cross_up(i):
            if (not np.isnan(curr_sma)) and (not np.isnan(curr_close)) and (curr_close > curr_sma):
                _IN_POSITION = True
                _ENTRY_INDEX = i
                _HIGHEST_SINCE_ENTRY = curr_high if not np.isnan(curr_high) else curr_close
                # Buy using 50% of equity
                return (0.5, 2, 1)

    # EXIT: have position and either MACD cross down or trailing stop triggered
    else:
        if macd_cross_down(i) or trailing_hit:
            _IN_POSITION = False
            _ENTRY_INDEX = None
            _HIGHEST_SINCE_ENTRY = -np.inf
            # Close entire long position
            return (-np.inf, 2, 1)

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
    Precompute MACD, ATR and SMA indicators using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are 1D numpy arrays of the same length as input ohlcv.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        high_s = ohlcv["close"].astype(float)
    else:
        high_s = ohlcv["high"].astype(float)

    close_s = ohlcv["close"].astype(float)

    # Low is optional in DATA_SCHEMA; fall back to close if missing
    if "low" in ohlcv.columns:
        low_s = ohlcv["low"].astype(float)
    else:
        low_s = close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.to_numpy().astype(float)
    signal_arr = macd_ind.signal.to_numpy().astype(float)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_arr = atr_ind.atr.to_numpy().astype(float)

    # Compute SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_arr = sma_ind.ma.to_numpy().astype(float)

    return {
        "close": close_s.to_numpy().astype(float),
        "high": high_s.to_numpy().astype(float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }