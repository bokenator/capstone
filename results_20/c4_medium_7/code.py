import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Module-level fallback to store last entry index when OrderContext doesn't expose it
__LAST_ENTRY_IDX = None


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
    """
    global __LAST_ENTRY_IDX

    i = int(c.i)
    pos = float(c.position_now)

    # Safety: bounds check
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Helper: safe access with NaN checks
    def is_finite(x):
        try:
            return np.isfinite(x)
        except Exception:
            return False

    # Detect MACD crossover up at index i
    def macd_cross_up(idx: int) -> bool:
        if idx == 0:
            return False
        a1, b1 = macd[idx - 1], signal[idx - 1]
        a2, b2 = macd[idx], signal[idx]
        if not (is_finite(a1) and is_finite(b1) and is_finite(a2) and is_finite(b2)):
            return False
        return (a1 <= b1) and (a2 > b2)

    # Detect MACD crossover down at index i
    def macd_cross_down(idx: int) -> bool:
        if idx == 0:
            return False
        a1, b1 = macd[idx - 1], signal[idx - 1]
        a2, b2 = macd[idx], signal[idx]
        if not (is_finite(a1) and is_finite(b1) and is_finite(a2) and is_finite(b2)):
            return False
        return (a1 >= b1) and (a2 < b2)

    # Try to obtain current position's entry index from OrderContext
    def get_entry_idx() -> int:
        # Prefer pos_record_now if available
        try:
            if hasattr(c, 'pos_record_now'):
                pr = c.pos_record_now
                # Try attribute access for entry index
                try:
                    entry_idx_candidate = int(pr.entry_idx)
                except Exception:
                    entry_idx_candidate = None

                if entry_idx_candidate is not None and entry_idx_candidate >= 0:
                    return entry_idx_candidate
        except Exception:
            pass

        # Fallback to last stored entry index (set when we place an entry order)
        try:
            if __LAST_ENTRY_IDX is not None:
                return int(__LAST_ENTRY_IDX)
        except Exception:
            pass

        return None

    # ENTRY: No position and MACD crosses above signal and price above SMA
    try:
        price = float(close[i])
    except Exception:
        price = np.nan

    sma_i = sma[i] if (i < len(sma)) else np.nan

    if pos == 0.0:
        if macd_cross_up(i) and is_finite(sma_i) and is_finite(price) and (price > sma_i):
            # Enter full long position (100% of equity)
            # Store expected entry index as fallback
            __LAST_ENTRY_IDX = i
            return (1.0, 2, 1)

        # No action
        return (np.nan, 0, 0)

    # HAVE POSITION: check for exit conditions
    # 1) MACD crosses below signal
    if macd_cross_down(i):
        # Clear fallback
        __LAST_ENTRY_IDX = None
        return (-np.inf, 2, 1)

    # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
    if not is_finite(atr[i]):
        # Can't compute trailing stop without ATR
        return (np.nan, 0, 0)

    entry_idx = get_entry_idx()
    if entry_idx is None:
        # No reliable entry index; skip trailing stop check
        return (np.nan, 0, 0)

    # Ensure indices valid
    entry_idx = int(entry_idx)
    if entry_idx < 0:
        return (np.nan, 0, 0)
    if entry_idx > i:
        # Entry in the future? skip
        return (np.nan, 0, 0)

    # Compute highest high since entry (inclusive)
    try:
        highest = np.nanmax(high[entry_idx:i + 1])
    except Exception:
        highest = np.nan

    if not is_finite(highest):
        return (np.nan, 0, 0)

    stop_level = highest - (trailing_mult * float(atr[i]))

    # If current close falls strictly below stop_level, exit
    if is_finite(price) and price < stop_level:
        __LAST_ENTRY_IDX = None
        return (-np.inf, 2, 1)

    return (np.nan, 0, 0)


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

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are numpy arrays of same length as input.
    """

    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' and 'high' columns")

    # Provide fallback for 'low' if not present (required for ATR)
    if 'low' not in ohlcv.columns:
        low_series = ohlcv['close']
    else:
        low_series = ohlcv['low']

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast,
                             slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }