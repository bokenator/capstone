"""
MACD + ATR trailing stop strategy helper functions for vectorbt backtests.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
    -> dict of numpy arrays: macd, signal, atr, sma, close, high

- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)
    -> returns a tuple (size, size_type, direction) where size can be np.nan for no order.

Notes:
- Does not use numba.
- Uses only pandas/numpy/vectorbt-compatible structures.

"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


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

    Returns a dictionary with numpy arrays for keys:
    - macd: MACD line
    - signal: MACD signal line
    - atr: Average True Range (Wilder's EMA)
    - sma: Simple Moving Average of close
    - close: Close prices
    - high: High prices

    The outputs align with the input index and have the same length as the input.
    """
    # Basic input validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise KeyError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD (using exponential moving averages)
    # Use ewm with adjust=False for no lookahead and stable behavior
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # ATR (True Range + Wilder's smoothing -> EMA with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing using ewm with alpha=1/atr_period
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # Convert to numpy arrays for performance and compatibility
    return {
        "macd": macd.to_numpy(dtype=float),
        "signal": signal.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "sma": sma.to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        "high": high.to_numpy(dtype=float),
    }


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """
    Order function implementing:
    - Entry: MACD crosses above signal AND price > 50-period SMA
    - Exit: MACD crosses below signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction) where:
    - size: float or np.nan (np.nan means no order)
    - size_type: int (enum value expected by vectorbt; see run_backtest wrapper)
    - direction: int (enum value expected by vectorbt)

    Notes:
    - This function avoids lookahead by only using data up to c.i.
    - It is defensive about missing data and warmup periods.
    """
    # Defensive conversions
    i = int(getattr(c, "i", 0))

    # Helper to return no order
    def no_order():
        return (np.nan, 0, 0)

    # If indicator arrays are shorter than i, return no order
    n = len(close)
    if i >= n:
        return no_order()

    # For crossover detection we need a previous bar
    if i == 0:
        prev_i = 0
    else:
        prev_i = i - 1

    # Protect against NaNs in indicators (warmup)
    try:
        macd_i = float(macd[i])
        signal_i = float(signal[i])
    except Exception:
        return no_order()

    try:
        macd_prev = float(macd[prev_i])
        signal_prev = float(signal[prev_i])
    except Exception:
        macd_prev = np.nan
        signal_prev = np.nan

    # Price and sma at current bar
    price_i = float(close[i])
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # Determine current position state
    pos = getattr(c, "position", None)
    in_position = False
    entry_idx = None

    # Try common attribute names for open position
    if pos is not None:
        # Many vectorbt versions provide c.position.is_open
        in_position = bool(getattr(pos, "is_open", False))
        # Try different possible names for entry index
        entry_idx = getattr(pos, "entry_i", None)
        if entry_idx is None:
            entry_idx = getattr(pos, "entry_idx", None)
        if entry_idx is None:
            entry_idx = getattr(pos, "entry_index", None)
        # If entry_idx is not None but not an int, try to coerce
        if entry_idx is not None:
            try:
                entry_idx = int(entry_idx)
            except Exception:
                entry_idx = None

    # ------------------------------------------------------------------
    # Entry logic
    # MACD crosses above signal and price above SMA
    macd_cross_up = False
    if not np.isnan(macd_prev) and not np.isnan(signal_prev):
        macd_cross_up = (macd_prev <= signal_prev) and (macd_i > signal_i)

    if (not in_position) and macd_cross_up and (not np.isnan(sma_i)) and (price_i > sma_i):
        # Open a long position: return size=1.0 as absolute amount, direction=Long
        # We return integers for size_type and direction. The run_backtest wrapper
        # will convert the tuple into a proper order. Common mapping in vectorbt
        # enums: SizeType.Amount=0, Direction.Long=1. These integers are used here
        # to avoid importing enums (the wrapper handles conversion).
        return (1.0, 0, 1)

    # ------------------------------------------------------------------
    # Exit logic (only when in position)
    if in_position:
        # MACD bearish cross
        macd_cross_down = False
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            macd_cross_down = (macd_prev >= signal_prev) and (macd_i < signal_i)

        # Compute highest price since entry (inclusive)
        # If entry_idx missing, fall back to scanning from start until current bar
        start_idx = 0 if entry_idx is None else max(0, entry_idx)
        # Clip to valid range
        start_idx = int(min(max(start_idx, 0), i))
        try:
            highest_since_entry = float(np.nanmax(high[start_idx : i + 1]))
        except Exception:
            highest_since_entry = np.nan

        atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan

        trailing_stop = np.nan
        if (not np.isnan(highest_since_entry)) and (not np.isnan(atr_i)):
            trailing_stop = highest_since_entry - float(trailing_mult) * atr_i

        # If MACD cross down OR price falls below trailing stop -> exit
        stop_hit = False
        if not np.isnan(trailing_stop):
            stop_hit = price_i < trailing_stop

        if macd_cross_down or stop_hit:
            # Close position by targeting size 0. Use SizeType.Target (commonly 2)
            # and Direction.Both (commonly 0) so that the position is closed.
            return (0.0, 2, 0)

    # No action
    return no_order()
