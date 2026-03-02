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
    Order function combining MACD crossover entries with ATR-based trailing stops.

    - Entry: MACD crosses above Signal AND price > SMA
    - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    This function maintains minimal persistent state on the function object to track
    the highest price since the most recent entry.

    Notes:
    - This is a plain Python function (no numba).
    - Returns a tuple (size, size_type, direction) as required by the backtest wrapper.
    """
    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, "position_now") else 0.0

    # Initialize persistent attributes on the function object on first call
    if not hasattr(order_func, "_initialized") or order_func._initialized is False:
        order_func._entry_idx = None  # index where current position was entered
        order_func._highest = -np.inf  # highest high since entry
        order_func._initialized = True

    # Helper to check safe numeric values
    def is_valid(x):
        return x is not None and (not np.isnan(x)) and np.isfinite(x)

    # Ensure arrays are numpy arrays
    close_arr = np.asarray(close)
    high_arr = np.asarray(high)
    macd_arr = np.asarray(macd)
    signal_arr = np.asarray(signal)
    atr_arr = np.asarray(atr)
    sma_arr = np.asarray(sma)

    # Bound-check index
    if i < 0 or i >= len(close_arr):
        return (np.nan, 0, 0)

    price = float(close_arr[i]) if is_valid(close_arr[i]) else np.nan

    # No position: check for entry
    if pos == 0 or np.isclose(pos, 0.0):
        # Reset tracking state
        order_func._entry_idx = None
        order_func._highest = -np.inf

        # Need at least one prior bar for crossover detection
        if i == 0:
            return (np.nan, 0, 0)

        # Validate indicators
        if not (is_valid(macd_arr[i]) and is_valid(signal_arr[i]) and is_valid(macd_arr[i - 1]) and is_valid(signal_arr[i - 1]) and is_valid(sma_arr[i])):
            return (np.nan, 0, 0)

        macd_cross_up = (macd_arr[i - 1] <= signal_arr[i - 1]) and (macd_arr[i] > signal_arr[i])
        price_above_sma = is_valid(price) and (price > float(sma_arr[i]))

        if macd_cross_up and price_above_sma:
            # Record entry index and initialize highest price tracker
            order_func._entry_idx = i
            # Use intrabar high to track the highest price since entry
            if is_valid(high_arr[i]):
                order_func._highest = float(high_arr[i])
            else:
                order_func._highest = price if is_valid(price) else -np.inf

            # Enter: use 50% of equity (Percent size_type = 2), long-only (direction = 1)
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # Have a (long) position: update highest and check for exits
    else:
        # If we don't have entry tracking info (e.g., resumed from external state), initialize it
        if order_func._entry_idx is None:
            order_func._entry_idx = i
            order_func._highest = float(high_arr[i]) if is_valid(high_arr[i]) else (price if is_valid(price) else -np.inf)

        # Update highest price since entry using today's high
        if is_valid(high_arr[i]):
            order_func._highest = max(order_func._highest, float(high_arr[i]))

        # Check for MACD bearish crossover
        macd_cross_down = False
        if i > 0 and is_valid(macd_arr[i]) and is_valid(signal_arr[i]) and is_valid(macd_arr[i - 1]) and is_valid(signal_arr[i - 1]):
            macd_cross_down = (macd_arr[i - 1] >= signal_arr[i - 1]) and (macd_arr[i] < signal_arr[i])

        # Check trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        trailing_hit = False
        if is_valid(atr_arr[i]) and is_valid(order_func._highest) and is_valid(price):
            stop_level = order_func._highest - float(trailing_mult) * float(atr_arr[i])
            trailing_hit = price < stop_level

        if macd_cross_down or trailing_hit:
            # Clear tracking state
            order_func._entry_idx = None
            order_func._highest = -np.inf
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
    Compute indicators required by the strategy using vectorbt indicators.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a 1D numpy array aligned with the input ohlcv index.
    """
    # Basic validation
    required_cols = {"high", "close"}
    missing = required_cols - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")

    # Ensure series are numeric floats
    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)

    # low may be missing; ATR requires low -> fall back to close if absent
    if "low" in ohlcv.columns:
        low_s = ohlcv["low"].astype(float)
    else:
        low_s = close_s.copy()

    # Compute MACD
    macd_res = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_res = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # Compute SMA
    sma_res = vbt.MA.run(close_s, window=sma_period)

    return {
        "close": close_s.values,
        "high": high_s.values,
        "macd": macd_res.macd.values,
        "signal": macd_res.signal.values,
        "atr": atr_res.atr.values,
        "sma": sma_res.ma.values,
    }