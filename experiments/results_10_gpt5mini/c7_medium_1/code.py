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
    trailing_mult: float,
) -> tuple:
    """
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    - Entry: MACD crosses above Signal AND price > 50-period SMA
    - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Notes:
    - This is a pure Python function (no numba).
    - Uses the provided `c` OrderContext to store per-position state (highest price since entry).

    Returns a tuple (size, size_type, direction) as required by the runner.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks for index bounds and NaNs
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # If key indicators are NaN at current bar, do nothing
    if np.isnan(macd[i]) or np.isnan(signal[i]) or np.isnan(sma[i]) or np.isnan(close[i]):
        return (np.nan, 0, 0)

    # Helper: detect cross up / down without raising on i==0
    def cross_up(arr1, arr2, idx):
        if idx == 0:
            return False
        if np.isnan(arr1[idx - 1]) or np.isnan(arr2[idx - 1]):
            return False
        return (arr1[idx] > arr2[idx]) and (arr1[idx - 1] <= arr2[idx - 1])

    def cross_down(arr1, arr2, idx):
        if idx == 0:
            return False
        if np.isnan(arr1[idx - 1]) or np.isnan(arr2[idx - 1]):
            return False
        return (arr1[idx] < arr2[idx]) and (arr1[idx - 1] >= arr2[idx - 1])

    # No position: check for entry
    if pos == 0.0:
        should_enter = cross_up(macd, signal, i) and (close[i] > sma[i])
        if should_enter:
            # Initialize trailing state on the context so it persists across bars
            try:
                setattr(c, "entry_idx", i)
                setattr(c, "entry_high", float(high[i]) if not np.isnan(high[i]) else float(close[i]))
            except Exception:
                # In case context doesn't allow setting attrs, ignore - still place the order
                pass

            # Enter full long position using 100% of equity (Percent size_type = 2)
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # Have a position: update highest_since_entry and check exits
    else:
        # Ensure ATR available
        if np.isnan(atr[i]):
            return (np.nan, 0, 0)

        # Update highest since entry on the context
        try:
            if not hasattr(c, "entry_high") or c.entry_high is None:
                # If missing (e.g., context restarted), initialize with current high
                setattr(c, "entry_high", float(high[i]) if not np.isnan(high[i]) else float(close[i]))
                setattr(c, "entry_idx", int(i))
            else:
                # Update the tracked high
                curr_high = float(high[i]) if not np.isnan(high[i]) else float(close[i])
                # Use max to track the highest observed price since entry
                if curr_high > float(c.entry_high):
                    setattr(c, "entry_high", curr_high)
        except Exception:
            # If we can't set attributes for some reason, proceed without persistent tracking
            pass

        # Compute trailing stop price using the tracked high
        tracked_high = getattr(c, "entry_high", float(high[i]) if not np.isnan(high[i]) else float(close[i]))
        trailing_stop_price = tracked_high - float(trailing_mult) * float(atr[i])

        # Exit conditions
        macd_bearish_cross = cross_down(macd, signal, i)
        price_below_trailing = (close[i] < trailing_stop_price) if not np.isnan(trailing_stop_price) else False

        if macd_bearish_cross or price_below_trailing:
            # Clear entry tracking (best-effort)
            try:
                setattr(c, "entry_high", None)
                setattr(c, "entry_idx", None)
            except Exception:
                pass

            # Close entire long position using Percent=-inf convention
            return (-np.inf, 2, 1)

        # Otherwise hold
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
    Compute indicators required by the strategy using vectorbt indicators.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    Each value is a 1-D numpy array aligned to the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("Input DataFrame must contain 'close' and 'high' columns as per DATA_SCHEMA")

    # Prepare series (ensure float dtype)
    close_s = ohlcv['close'].astype(float)
    high_s = ohlcv['high'].astype(float)

    # Low may be missing - if so, fallback to close (conservative)
    if 'low' in ohlcv.columns:
        low_s = ohlcv['low'].astype(float)
    else:
        low_s = close_s.copy()

    # Run indicators using vectorbt
    macd_res = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_res = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    sma_res = vbt.MA.run(close_s, window=sma_period)

    macd_arr = macd_res.macd.values
    signal_arr = macd_res.signal.values
    atr_arr = atr_res.atr.values
    sma_arr = sma_res.ma.values

    # Forward-fill internal NaNs (safe, does not introduce lookahead), but keep initial NaNs
    # so that signals won't fire until enough data is present.
    macd_arr = pd.Series(macd_arr).ffill().values
    signal_arr = pd.Series(signal_arr).ffill().values
    atr_arr = pd.Series(atr_arr).ffill().values
    sma_arr = pd.Series(sma_arr).ffill().values

    return {
        'close': close_s.values,
        'high': high_s.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }