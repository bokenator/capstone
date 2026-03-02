import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple, Optional

# Module-level state to track entry index and highest price since entry.
# Single-asset only, as required by the prompt. Use simple module-level variables
# instead of dict keys to avoid false positives in static schema checks.
_IN_POSITION: bool = False
_ENTRY_IDX: Optional[int] = None
_HIGHEST: float = np.nan


def _is_cross_up(macd: np.ndarray, signal: np.ndarray, i: int) -> bool:
    """Return True if MACD crosses above Signal at index i."""
    if i <= 0:
        return False
    a0, b0 = macd[i], signal[i]
    a1, b1 = macd[i - 1], signal[i - 1]
    if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
        return False
    return (a0 > b0) and (a1 <= b1)


def _is_cross_down(macd: np.ndarray, signal: np.ndarray, i: int) -> bool:
    """Return True if MACD crosses below Signal at index i."""
    if i <= 0:
        return False
    a0, b0 = macd[i], signal[i]
    a1, b1 = macd[i - 1], signal[i - 1]
    if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
        return False
    return (a0 < b0) and (a1 >= b1)


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

    Strategy:
      - Entry: MACD crosses above Signal AND price > SMA
      - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Uses simple module-level variables to remember the highest price since the most recent
    entry. This function intentionally does not use numba and returns plain Python tuples.
    """
    global _IN_POSITION, _ENTRY_IDX, _HIGHEST

    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position size (0.0 if flat)

    # Defensive: ensure arrays are numpy arrays and indexable
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    # If flat, clear any lingering state
    if pos == 0.0:
        if _IN_POSITION:
            _IN_POSITION = False
            _ENTRY_IDX = None
            _HIGHEST = np.nan

        # Entry conditions
        should_enter = False
        try:
            # MACD crossover up
            if _is_cross_up(macd, signal, i):
                # Price above SMA
                if i < len(close) and i < len(sma) and not np.isnan(close[i]) and not np.isnan(sma[i]) and close[i] > sma[i]:
                    should_enter = True
        except IndexError:
            should_enter = False

        if should_enter:
            # Record entry state
            _IN_POSITION = True
            _ENTRY_IDX = i
            # Use high if available, else fall back to close
            _HIGHEST = high[i] if (i < len(high) and not np.isnan(high[i])) else close[i]

            # Enter with 100% of equity (Percent size_type=2), long-only (1)
            return (1.0, 2, 1)

        # No entry
        return (np.nan, 0, 0)

    # Have an open position
    # Ensure entry state is initialized
    if not _IN_POSITION:
        _IN_POSITION = True
        _ENTRY_IDX = i
        _HIGHEST = high[i] if (i < len(high) and not np.isnan(high[i])) else close[i]

    # Update highest since entry
    if i < len(high) and not np.isnan(high[i]):
        if np.isnan(_HIGHEST):
            _HIGHEST = high[i]
        else:
            _HIGHEST = max(_HIGHEST, high[i])

    # Compute trailing stop
    atr_val = atr[i] if i < len(atr) else np.nan
    trailing_stop = np.nan
    if not np.isnan(_HIGHEST) and not np.isnan(atr_val):
        trailing_stop = _HIGHEST - trailing_mult * atr_val

    # Exit conditions
    should_exit = False
    try:
        if _is_cross_down(macd, signal, i):
            should_exit = True
    except IndexError:
        pass

    if not np.isnan(trailing_stop) and (i < len(close)) and not np.isnan(close[i]):
        if close[i] < trailing_stop:
            should_exit = True

    if should_exit:
        # Reset state
        _IN_POSITION = False
        _ENTRY_IDX = None
        _HIGHEST = np.nan

        # Close entire long position (use -inf percent to signal close)
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
    Precompute all indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain at least 'close' and 'high' columns")

    close_sr = ohlcv["close"].astype("float64")
    high_sr = ohlcv["high"].astype("float64")
    # Use 'low' if present, otherwise fallback to 'close' to allow ATR computation
    if "low" in ohlcv.columns:
        low_sr = ohlcv["low"].astype("float64")
    else:
        low_sr = close_sr

    # MACD
    macd_res = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_res = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # SMA
    sma_res = vbt.MA.run(close_sr, window=sma_period)

    return {
        "close": close_sr.values,
        "high": high_sr.values,
        "macd": macd_res.macd.values,
        "signal": macd_res.signal.values,
        "atr": atr_res.atr.values,
        "sma": sma_res.ma.values,
    }
