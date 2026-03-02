import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


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
    Order function for vectorbt.from_order_func (non-numba).

    Implements MACD crossover entries filtered by a SMA trend and
    ATR-based trailing stop exits.

    State (entry index and highest price since entry) is tracked using
    function attributes and reset at the start of each run (c.i == 0).

    Returns a tuple (size, size_type, direction) as required by the
    backtest wrapper.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize/reset state at the beginning of a run
    if i == 0:
        # _entry_idx: index where current position was entered (or -1 if flat)
        # _highest: highest high observed since entry
        order_func._entry_idx = -1
        order_func._highest = -np.inf

    # Helper to check safe access of previous bar
    def _has_prev(idx: int) -> bool:
        return idx - 1 >= 0

    # Ensure inputs are numpy arrays
    close_arr = np.asarray(close, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    macd_arr = np.asarray(macd, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    atr_arr = np.asarray(atr, dtype=float)
    sma_arr = np.asarray(sma, dtype=float)

    # Entry logic (only when flat)
    if pos == 0.0:
        # Safe checks for NaNs and previous values
        can_check_cross = _has_prev(i) and not (np.isnan(macd_arr[i]) or np.isnan(signal_arr[i]) or np.isnan(macd_arr[i - 1]) or np.isnan(signal_arr[i - 1]))
        sma_ok = not np.isnan(sma_arr[i]) and close_arr[i] > sma_arr[i]

        should_enter = False
        if can_check_cross and sma_ok:
            # MACD line crosses above signal
            if (macd_arr[i] > signal_arr[i]) and (macd_arr[i - 1] <= signal_arr[i - 1]):
                should_enter = True

        if should_enter:
            # Record entry index and initial highest price
            order_func._entry_idx = i
            # Initialize highest with current bar high (use available high value)
            current_high = high_arr[i] if not np.isnan(high_arr[i]) else close_arr[i]
            order_func._highest = float(current_high)

            # Use 100% of equity to enter long
            return (1.0, 2, 1)

    else:
        # We have a position - update highest_since_entry
        # If entry_idx not set (shouldn't happen), set it to first bar with position
        if getattr(order_func, "_entry_idx", -1) == -1:
            order_func._entry_idx = i
            order_func._highest = high_arr[i] if not np.isnan(high_arr[i]) else close_arr[i]

        # Update highest price seen since entry using current high
        if not np.isnan(high_arr[i]):
            if np.isnan(order_func._highest):
                order_func._highest = float(high_arr[i])
            else:
                order_func._highest = float(max(order_func._highest, high_arr[i]))

        # Compute trailing stop level if ATR available
        stop_price = np.nan
        if not np.isnan(atr_arr[i]) and not np.isinf(order_func._highest):
            stop_price = order_func._highest - float(trailing_mult) * float(atr_arr[i])

        # MACD bearish cross check
        macd_bear_cross = False
        if _has_prev(i) and not (np.isnan(macd_arr[i]) or np.isnan(signal_arr[i]) or np.isnan(macd_arr[i - 1]) or np.isnan(signal_arr[i - 1])):
            if (macd_arr[i] < signal_arr[i]) and (macd_arr[i - 1] >= signal_arr[i - 1]):
                macd_bear_cross = True

        trailing_stop_hit = False
        if not np.isnan(stop_price):
            # If price falls strictly below the stop, trigger exit
            if not np.isnan(close_arr[i]) and close_arr[i] < stop_price:
                trailing_stop_hit = True

        if macd_bear_cross or trailing_stop_hit:
            # Reset entry tracking for safety
            order_func._entry_idx = -1
            order_func._highest = -np.inf

            # Close entire long position
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
    Compute MACD, Signal, ATR and SMA indicators using vectorbt.

    Returns a dict with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    Edge cases:
    - If 'low' column is missing, use 'close' as a fallback for ATR computation.
    - Forward-fill (ffill) indicator NaNs to avoid transient gaps after initial warmup.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' and 'high' columns")

    close_s = ohlcv['close'].astype(float)
    high_s = ohlcv['high'].astype(float)
    low_s = ohlcv['low'].astype(float) if 'low' in ohlcv.columns else close_s

    # Compute MACD using vectorbt
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR using vectorbt (requires high, low, close)
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # Compute SMA for trend filter
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    # Extract numpy arrays
    close_arr = close_s.to_numpy(dtype=float)
    high_arr = high_s.to_numpy(dtype=float)
    macd_arr = macd_ind.macd.to_numpy(dtype=float)
    signal_arr = macd_ind.signal.to_numpy(dtype=float)
    atr_arr = atr_ind.atr.to_numpy(dtype=float)
    sma_arr = sma_ind.ma.to_numpy(dtype=float)

    # Forward-fill NaNs to remove intermittent NaNs after initial warmup (no lookahead)
    def _ffill(a: np.ndarray) -> np.ndarray:
        s = pd.Series(a)
        s = s.ffill()
        return s.to_numpy(dtype=float)

    macd_arr = _ffill(macd_arr)
    signal_arr = _ffill(signal_arr)
    atr_arr = _ffill(atr_arr)
    sma_arr = _ffill(sma_arr)

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }