# Complete strategy implementation combining MACD crossover entries with ATR-based trailing stops.
# Exports:
# - compute_indicators
# - order_func

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import vectorbt as vbt


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dictionary with the following keys (1D numpy arrays):
      - macd: MACD line
      - signal: MACD signal line
      - atr: Average True Range
      - sma: Simple moving average (period = sma_period)
      - close: close prices
      - high: high prices

    Args:
        ohlcv: DataFrame with at least columns ['open','high','low','close'] (volume optional)
        macd_fast, macd_slow, macd_signal: MACD parameters
        sma_period: period for trend filter SMA
        atr_period: ATR window
    """

    # Validate input
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close_series = ohlcv["close"].astype(float)
    high_series = ohlcv["high"].astype(float)
    low_series = ohlcv["low"].astype(float)

    # MACD
    # vbt.MACD.run returns an object with attributes `macd` and `signal` (pandas Series)
    macd_res = vbt.MACD.run(
        close_series,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR
    # vbt.ATR.run returns an object with attributes `tr` and `atr` (pandas Series)
    atr_res = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (using MA indicator)
    ma_res = vbt.MA.run(close_series, window=sma_period, ewm=False)

    # Extract 1D numpy arrays. Use .values to preserve the original index order.
    # For safety, flatten any accidental 2D shape.
    def _to_1d(arr_like: Any) -> np.ndarray:
        arr = np.asarray(arr_like)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr

    macd_arr = _to_1d(macd_res.macd.values)
    signal_arr = _to_1d(macd_res.signal.values)
    atr_arr = _to_1d(atr_res.atr.values)
    sma_arr = _to_1d(ma_res.ma.values)
    close_arr = _to_1d(close_series.values)
    high_arr = _to_1d(high_series.values)

    # Ensure float dtype
    macd_arr = macd_arr.astype(float)
    signal_arr = signal_arr.astype(float)
    atr_arr = atr_arr.astype(float)
    sma_arr = sma_arr.astype(float)
    close_arr = close_arr.astype(float)
    high_arr = high_arr.astype(float)

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": close_arr,
        "high": high_arr,
    }


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
    Order function implementing the strategy logic.

    The function is called sequentially by vectorbt. We maintain minimal state on the
    function object to track whether we're currently in a long position, when it was
    entered, and the highest price seen since entry (for trailing stop calculations).

    Returns a tuple (size, size_type, direction):
      - To enter a long position: (1.0, SizeType.TargetPercent (5), Direction.LongOnly (0))
        which targets 100% of portfolio value to the asset.
      - To exit: (0.0, SizeType.TargetPercent (5), Direction.LongOnly (0))
      - No action: (np.nan, 0, 0)

    Notes:
      - Uses MACD crossover for entries/exits and an ATR-based trailing stop.
      - Long-only strategy.
      - Does not use numba.
    """

    # Initialize persistent state on the function object
    if not hasattr(order_func, "_state"):
        order_func._state = {
            "in_position": False,
            "entry_index": None,
            "highest_price": np.nan,
        }

    state = order_func._state

    # Get current bar index
    i = getattr(c, "i", None)
    if i is None:
        # If context index not available, do nothing
        return (np.nan, 0, 0)

    # Defensive conversions
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    # Helper to safely get previous values
    def _get_prev(arr: np.ndarray, idx: int) -> float:
        if idx <= 0:
            return np.nan
        return float(arr[idx - 1])

    # Current and previous MACD/Signal
    cur_macd = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    cur_signal = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    prev_macd = _get_prev(macd, i)
    prev_signal = _get_prev(signal, i)

    # Detect crossovers (require previous values to be non-NaN)
    macd_cross_up = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and not np.isnan(cur_macd)
        and not np.isnan(cur_signal)
        and (prev_macd <= prev_signal)
        and (cur_macd > cur_signal)
    )

    macd_cross_down = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and not np.isnan(cur_macd)
        and not np.isnan(cur_signal)
        and (prev_macd >= prev_signal)
        and (cur_macd < cur_signal)
    )

    # Price vs SMA filter
    price = float(close[i]) if not np.isnan(close[i]) else np.nan
    sma_val = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    price_above_sma = (not np.isnan(price)) and (not np.isnan(sma_val)) and (price > sma_val)

    # Entry logic: MACD cross up + price above SMA, only if not already in position
    if (not state["in_position"]) and macd_cross_up and price_above_sma:
        # Enter: target 100% long
        state["in_position"] = True
        state["entry_index"] = int(i)
        # Set highest price since entry to current high (fallback to close)
        hp = float(high[i]) if not np.isnan(high[i]) else price
        state["highest_price"] = hp
        # SizeType.TargetPercent = 5, Direction.LongOnly = 0
        return (1.0, 5, 0)

    # If in position, update highest price and check exit conditions
    if state["in_position"]:
        # Update highest price since entry
        current_high = float(high[i]) if not np.isnan(high[i]) else price
        if np.isnan(state["highest_price"]) or current_high > state["highest_price"]:
            state["highest_price"] = current_high

        # Compute trailing stop level if ATR available
        atr_val = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        hp = state["highest_price"]
        stop_level = hp - (trailing_mult * atr_val) if (not np.isnan(hp) and not np.isnan(atr_val)) else np.nan

        price_below_stop = (not np.isnan(price)) and (not np.isnan(stop_level)) and (price < stop_level)

        # Exit on MACD cross down OR price falling below trailing stop
        if macd_cross_down or price_below_stop:
            # Exit: target 0% long
            state["in_position"] = False
            state["entry_index"] = None
            state["highest_price"] = np.nan
            return (0.0, 5, 0)

    # Default: no action
    return (np.nan, 0, 0)
