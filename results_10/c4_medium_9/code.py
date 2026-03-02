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

    This implementation uses function attributes to keep simple per-run state
    (entry index, highest price since entry, and pending entry flag).

    Entry: MACD crosses above Signal AND close > SMA
    Exit:  MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    Returns tuple (size, size_type, direction) or (np.nan, 0, 0) for no action.
    Uses 50%% of equity for entries (size_type=2, size=0.5) and -np.inf percent to close.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent state on the function object (per-Python-process run)
    if not hasattr(order_func, "_entry_idx"):
        order_func._entry_idx = None  # type: ignore[attr-defined]
        order_func._entry_high = np.nan  # type: ignore[attr-defined]
        order_func._pending_entry = False  # type: ignore[attr-defined]

    # Sync pending flag with actual position: if position is open, pending_entry should be False
    if pos != 0 and getattr(order_func, "_pending_entry", False):
        order_func._pending_entry = False  # entry was executed

    # Helper to safely check NaNs for index
    def is_valid_idx(arr: np.ndarray, idx: int) -> bool:
        try:
            return not np.isnan(arr[idx])
        except Exception:
            return False

    # ENTRY: only when flat
    if pos == 0:
        # If we previously placed an entry order but it's not executed yet, do not place another
        if getattr(order_func, "_pending_entry", False):
            return (np.nan, 0, 0)

        # Need at least one previous bar to detect crossover
        if i == 0:
            return (np.nan, 0, 0)

        # Validate required values
        if not (is_valid_idx(macd, i) and is_valid_idx(signal, i) and is_valid_idx(macd, i - 1) and is_valid_idx(signal, i - 1) and is_valid_idx(sma, i) and is_valid_idx(close, i)):
            return (np.nan, 0, 0)

        macd_now = float(macd[i])
        signal_now = float(signal[i])
        macd_prev = float(macd[i - 1])
        signal_prev = float(signal[i - 1])
        sma_now = float(sma[i])
        close_now = float(close[i])

        macd_cross_up = (macd_now > signal_now) and (macd_prev <= signal_prev)
        price_above_sma = close_now > sma_now

        if macd_cross_up and price_above_sma:
            # Place entry with 50% of equity (percent)
            # Set pending flag and initialize entry tracking state
            order_func._pending_entry = True
            order_func._entry_idx = i
            # Initialize highest price since entry with current high if available else close
            entry_high = high[i] if is_valid_idx(high, i) else close_now
            # Ensure numeric
            order_func._entry_high = float(entry_high) if not np.isnan(entry_high) else float(close_now)

            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # In-position: check exit conditions
    else:
        # If we don't have an entry index stored (e.g., started in position), initialize it to current bar
        if order_func._entry_idx is None:
            order_func._entry_idx = i
            order_func._entry_high = high[i] if is_valid_idx(high, i) else close[i]

        # Update highest_since_entry using current high
        if is_valid_idx(high, i):
            if np.isnan(order_func._entry_high):
                order_func._entry_high = float(high[i])
            else:
                order_func._entry_high = float(max(order_func._entry_high, float(high[i])))

        # Prepare current and previous MACD/Signal for crossover exit
        macd_cross_down = False
        if i > 0 and is_valid_idx(macd, i) and is_valid_idx(signal, i) and is_valid_idx(macd, i - 1) and is_valid_idx(signal, i - 1):
            macd_now = float(macd[i])
            signal_now = float(signal[i])
            macd_prev = float(macd[i - 1])
            signal_prev = float(signal[i - 1])
            macd_cross_down = (macd_now < signal_now) and (macd_prev >= signal_prev)

        # Trailing stop: price falls below (highest_since_entry - trailing_mult * atr_now)
        trail_trigger = False
        if not np.isnan(order_func._entry_high) and is_valid_idx(atr, i) and is_valid_idx(close, i):
            atr_now = float(atr[i])
            if not np.isnan(atr_now):
                stop_level = order_func._entry_high - float(trailing_mult) * atr_now
                trail_trigger = float(close[i]) < stop_level

        # Exit if any exit condition met
        if macd_cross_down or trail_trigger:
            # Reset tracking state
            order_func._entry_idx = None
            order_func._entry_high = np.nan
            order_func._pending_entry = False
            # Close entire long position (use -inf percent to signal full close)
            return (-np.inf, 2, 1)

        # Otherwise, no action
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
    Precompute indicators required by the strategy using vectorbt indicators.

    Returns a dictionary with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate input columns (only use columns declared in DATA_SCHEMA)
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_series = ohlcv['close'].astype(float)
    high_series = ohlcv['high'].astype(float)

    # Use 'low' if present, otherwise fall back to 'close' to allow ATR computation
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low'].astype(float)
    else:
        low_series = close_series

    # Run MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # Run ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_arr = atr_ind.atr.values

    # Run SMA (MA)
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }