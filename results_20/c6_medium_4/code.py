# -*- coding: utf-8 -*-
"""
MACD + ATR trailing stop strategy implementation for vectorbt backtester.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
    -> dict[str, np.ndarray]
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple[size, size_type, direction]

Notes:
- Does not use numba or vbt.portfolio.nb / vbt.portfolio.enums in this module.
- Uses TargetPercent (code 5) for sizing: entry sets target percent to 100% (1.0),
  exit sets target percent to 0.0.

References:
- MACD computed with EMA (ewm, adjust=False) so no lookahead.
- ATR computed using Wilder smoothing (alpha = 1 / period).

"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Global state to track highest price since entry per column.
# Key: column index (int), Value: float highest high seen since position opened
_HIGHEST_SINCE_ENTRY: Dict[int, float] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute MACD, Signal, SMA, ATR and expose close/high arrays.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least 'high','low','close').
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line EMA period for MACD.
        sma_period: Period for trend SMA filter.
        atr_period: Period for ATR.

    Returns:
        Dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'
        Each value is a 1-d numpy.ndarray with the same length as input.
    """
    # Basic validation and extraction
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    # Work on float copies to avoid modifying user data
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    # MACD: EMA fast - EMA slow, signal = EMA(macd)
    # Use ewm with adjust=False to be recursive (no lookahead)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA: simple moving average (use min_periods=sma_period to produce NaN until enough data)
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR: True Range then Wilder smoothing (alpha = 1/atr_period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing: ewm with alpha=1/atr_period and adjust=False
    if atr_period > 0:
        atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()
    else:
        # fallback: simple rolling mean if invalid period
        atr = tr.rolling(window=max(1, atr_period)).mean()

    # Convert to numpy arrays
    out = {
        "macd": macd.to_numpy(copy=True),
        "signal": signal.to_numpy(copy=True),
        "atr": atr.to_numpy(copy=True),
        "sma": sma.to_numpy(copy=True),
        "close": close.to_numpy(copy=True),
        "high": high.to_numpy(copy=True),
    }

    return out


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
    Order function for vectorbt.from_order_func (use_numba=False).

    Args:
        c: Order context provided by vectorbt. Expects attributes `i` (int index),
           `col` (int column index) and `position_now` (float current position size).
        close, high, macd, signal, atr, sma: 1-d numpy arrays of indicators (same length).
        trailing_mult: Multiplier for ATR to compute trailing stop distance (e.g., 2.0).

    Returns:
        Tuple (size, size_type, direction)
        - Uses SizeType.TargetPercent (code 5) to set target exposure: 1.0 enter, 0.0 exit.
        - Uses Direction.LongOnly (code 0) for long-only orders.
        - When no action, returns (np.nan, 0, 2) which is translated to a NoOrder by the wrapper.

    Important:
        - Maintains per-column highest price since entry in module-level dict _HIGHEST_SINCE_ENTRY.
        - Resets state at the beginning of a run (when c.i == 0).
    """
    # Defensive typing
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Reset global state at the start of the simulation (when first index is processed)
    if i == 0:
        _HIGHEST_SINCE_ENTRY.clear()

    # Safely get indicator values for current index
    # If arrays are shorter than i, raise to avoid silent errors
    try:
        curr_close = float(close[i])
        curr_high = float(high[i])
        curr_macd = float(macd[i])
        curr_signal = float(signal[i])
        curr_atr = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        curr_sma = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    except IndexError:
        # In case of mismatch, do nothing
        return (float("nan"), 0, 2)

    # Current position (0 means flat). position_now is provided by the context.
    pos_now = float(getattr(c, "position_now", 0.0))
    in_pos = pos_now != 0.0

    # If in position, update highest seen since entry
    if in_pos:
        prev_high = _HIGHEST_SINCE_ENTRY.get(col, curr_high)
        _HIGHEST_SINCE_ENTRY[col] = max(prev_high, curr_high)

    # Determine MACD crosses using only past & present values (no future)
    macd_cross_up = False
    macd_cross_down = False
    if i > 0:
        prev_macd = float(macd[i - 1])
        prev_signal = float(signal[i - 1])
        macd_cross_up = (prev_macd < prev_signal) and (curr_macd > curr_signal)
        macd_cross_down = (prev_macd > prev_signal) and (curr_macd < curr_signal)

    # Trailing stop condition: price falls below (highest_since_entry - trailing_mult * ATR)
    trailing_trigger = False
    if in_pos and (col in _HIGHEST_SINCE_ENTRY) and (not np.isnan(curr_atr)):
        highest = _HIGHEST_SINCE_ENTRY[col]
        stop_level = highest - float(trailing_mult) * curr_atr
        trailing_trigger = curr_close < stop_level

    # EXIT conditions (if currently in position): MACD bearish cross OR trailing stop
    if in_pos and (macd_cross_down or trailing_trigger):
        # Clear stored highest for this column
        _HIGHEST_SINCE_ENTRY.pop(col, None)
        # Use TargetPercent (code 5) to set target exposure to 0.0 -> close position
        size = 0.0
        size_type = 5  # TargetPercent
        direction = 0  # LongOnly
        return (size, size_type, direction)

    # ENTRY conditions (all must be true): MACD crosses above signal and price > 50-period SMA
    if (not in_pos) and macd_cross_up and (not np.isnan(curr_sma)) and (curr_close > curr_sma):
        # Initialize highest_since_entry with current high
        _HIGHEST_SINCE_ENTRY[col] = curr_high
        # Set target exposure to 100%
        size = 1.0
        size_type = 5  # TargetPercent
        direction = 0  # LongOnly
        return (size, size_type, direction)

    # No action
    return (float("nan"), 0, 2)
