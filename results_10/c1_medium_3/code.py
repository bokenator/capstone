import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any


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

    This function is intentionally pure-Python (NO NUMBA).

    Args:
        c: vectorbt OrderContext with attributes c.i, c.position_now, c.cash_now
        close: Close prices array
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        Tuple (size, size_type, direction) per the ORDER_CONTEXT_SCHEMA.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent attributes on the function to track entry state.
    # These are used to compute the highest price since the actual entry.
    if not hasattr(order_func, "_entry_idx"):
        order_func._entry_idx = None  # type: ignore
        order_func._highest_since_entry = np.nan  # type: ignore

    def _is_finite(val: Any) -> bool:
        try:
            return not np.isnan(val)
        except Exception:
            return False

    def is_cross_up(idx: int) -> bool:
        # Need at least one prior bar to detect a cross
        if idx <= 0 or idx >= len(macd):
            return False
        if not (_is_finite(macd[idx]) and _is_finite(signal[idx]) and _is_finite(macd[idx - 1]) and _is_finite(signal[idx - 1])):
            return False
        return (macd[idx - 1] <= signal[idx - 1]) and (macd[idx] > signal[idx])

    def is_cross_down(idx: int) -> bool:
        if idx <= 0 or idx >= len(macd):
            return False
        if not (_is_finite(macd[idx]) and _is_finite(signal[idx]) and _is_finite(macd[idx - 1]) and _is_finite(signal[idx - 1])):
            return False
        return (macd[idx - 1] >= signal[idx - 1]) and (macd[idx] < signal[idx])

    # If flat: look for entry
    if pos == 0.0:
        # Reset tracked entry state while flat
        order_func._entry_idx = None  # type: ignore
        order_func._highest_since_entry = np.nan  # type: ignore

        # Entry: MACD crosses above signal AND price above SMA
        if is_cross_up(i) and _is_finite(sma[i]) and _is_finite(close[i]) and (close[i] > sma[i]):
            # Use 100% of equity to enter long (Percent sizing)
            return (1.0, 2, 1)

        return (np.nan, 0, 0)  # No action

    # If in position: ensure we have an entry index and track highest price since entry
    # Recover entry index if missing by scanning backwards for the last valid entry signal
    if order_func._entry_idx is None:
        recovered_idx = None
        # Find the most recent bar (<= i) where an entry condition was True
        for idx in range(i, -1, -1):
            if is_cross_up(idx) and _is_finite(sma[idx]) and _is_finite(close[idx]) and (close[idx] > sma[idx]):
                recovered_idx = idx
                break
        # If not found, fall back to the current bar as conservative default
        if recovered_idx is None:
            recovered_idx = i
        order_func._entry_idx = int(recovered_idx)  # type: ignore
        # Initialize highest since entry
        try:
            order_func._highest_since_entry = float(np.nanmax(high[order_func._entry_idx : i + 1]))  # type: ignore
        except Exception:
            # Fallback to current high
            order_func._highest_since_entry = float(high[i]) if _is_finite(high[i]) else np.nan  # type: ignore

    # Update highest since entry with the current high
    if _is_finite(high[i]):
        if np.isnan(order_func._highest_since_entry):
            order_func._highest_since_entry = float(high[i])  # type: ignore
        else:
            order_func._highest_since_entry = float(max(order_func._highest_since_entry, high[i]))  # type: ignore

    # Determine exit conditions
    should_exit = False

    # Exit 1: MACD crosses below signal
    if is_cross_down(i):
        should_exit = True

    # Exit 2: Price falls below (highest_since_entry - trailing_mult * ATR)
    atr_ok = _is_finite(atr[i])
    highest_ok = not np.isnan(order_func._highest_since_entry)
    if atr_ok and highest_ok:
        trailing_level = order_func._highest_since_entry - float(trailing_mult) * float(atr[i])  # type: ignore
        if _is_finite(close[i]) and (close[i] < trailing_level):
            should_exit = True

    if should_exit:
        # Close entire long position by specifying negative amount equal to current position size
        size = -float(pos) if pos != 0.0 else 0.0
        # Reset tracked state; it will be re-initialized when flat
        order_func._entry_idx = None  # type: ignore
        order_func._highest_since_entry = np.nan  # type: ignore

        if size == 0.0:
            return (np.nan, 0, 0)
        return (size, 0, 1)

    # Otherwise, hold
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
    Precompute MACD, ATR, and SMA indicators required by the strategy.

    Uses vectorbt indicator wrappers to ensure consistent output lengths and indices.

    Args:
        ohlcv: DataFrame with columns at least ['high', 'close']. 'low' is optional.
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are numpy arrays aligned with the input index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # Use 'low' if available; otherwise fall back to 'close' to allow ATR computation
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low']
    else:
        # Fallback: use close as a conservative estimate for low
        low_series = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }
