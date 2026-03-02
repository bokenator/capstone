# Trading strategy: MACD crossover entries with ATR-based trailing stops
# Exports:
# - compute_indicators: compute MACD, signal, SMA, ATR and return numpy arrays
# - order_func: order function used by vectorbt.from_order_func (no numba)

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------------------

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

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least 'high','low','close').
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for trend SMA filter.
        atr_period: Period for ATR.

    Returns:
        Dictionary with numpy arrays: 'macd', 'signal', 'sma', 'atr', 'close', 'high'.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ("high", "low", "close"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)
    low_sr = ohlcv["low"].astype(float)

    # MACD (classic): EMA(fast) - EMA(slow), signal is EMA(macd, signal)
    # Use pandas ewm with adjust=False (no lookahead)
    ema_fast = close_sr.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_sr.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close_sr.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR: True Range and rolling mean (Wilder-style could be used but simple rolling is fine)
    prev_close = close_sr.shift(1)
    tr1 = high_sr - low_sr
    tr2 = (high_sr - prev_close).abs()
    tr3 = (low_sr - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    # Convert to numpy arrays
    macd_arr = macd_line.values.astype(float)
    signal_arr = signal_line.values.astype(float)
    sma_arr = sma.values.astype(float)
    atr_arr = atr.values.astype(float)
    close_arr = close_sr.values.astype(float)
    high_arr = high_sr.values.astype(float)

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "sma": sma_arr,
        "atr": atr_arr,
        "close": close_arr,
        "high": high_arr,
    }


# ---------------------------------------------------------------------------
# Order function (stateful across the run but reset at the first bar)
# ---------------------------------------------------------------------------

# Internal state used by order_func. It will be reset whenever c.i == 0 to
# ensure deterministic behavior across separate backtest runs / truncated data.
_order_state: Dict[str, Any] = {
    "in_position": False,
    "entry_idx": -1,
    "entry_high": np.nan,
}

# Constants for order tuple: (size, size_type, direction)
# We avoid importing vbt enums directly (runner will convert ints to enums).
SIZE_TYPE_AMOUNT = 0
DIRECTION_BOTH = 0
DIRECTION_LONG = 1
DIRECTION_SHORT = 2


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
    Order function for vectorbt.from_order_func.

    Arguments correspond to arrays passed by the backtester. The function
    decides on market orders based only on data up to and including the
    current index c.i (no lookahead).

    Returns a tuple (size, size_type, direction) or an Order-like object. We
    return tuples here; the run harness wraps this into vectorbt Order.

    Size semantics:
      - Entry (buy): (1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)
      - Exit  (sell): (1.0, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
      - No action: (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
    """
    global _order_state

    # Current index
    idx = int(getattr(c, "i", 0))

    # Reset state at start of run to avoid leakage between runs or truncated data
    if idx == 0:
        _order_state = {"in_position": False, "entry_idx": -1, "entry_high": np.nan}

    # Defensive bounds check
    n = len(close)
    if idx < 0 or idx >= n:
        # No action if index out of bounds
        return (float("nan"), SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Read current values (use local variables to avoid repeated indexing)
    close_price = float(close[idx])
    high_price = float(high[idx])

    # Safely read MACD / signal / ATR / SMA values (may be NaN early on)
    macd_val = float(macd[idx]) if not np.isnan(macd[idx]) else np.nan
    signal_val = float(signal[idx]) if not np.isnan(signal[idx]) else np.nan
    atr_val = float(atr[idx]) if not np.isnan(atr[idx]) else np.nan
    sma_val = float(sma[idx]) if not np.isnan(sma[idx]) else np.nan

    # Cross detection uses previous bar values (no lookahead)
    cross_up = False
    cross_down = False
    if idx > 0:
        prev_macd = float(macd[idx - 1]) if not np.isnan(macd[idx - 1]) else np.nan
        prev_signal = float(signal[idx - 1]) if not np.isnan(signal[idx - 1]) else np.nan
        if (not np.isnan(macd_val)) and (not np.isnan(signal_val)) and (not np.isnan(prev_macd)) and (not np.isnan(prev_signal)):
            cross_up = (macd_val > signal_val) and (prev_macd <= prev_signal)
            cross_down = (macd_val < signal_val) and (prev_macd >= prev_signal)

    # If not currently in a position, check entry conditions
    if not _order_state["in_position"]:
        enter = False
        # All must be true: MACD cross up and price above SMA
        if cross_up:
            # Trend filter: price (close) must be above SMA
            if not np.isnan(sma_val) and close_price > sma_val:
                enter = True
        if enter:
            # Initialize entry tracking
            _order_state["in_position"] = True
            _order_state["entry_idx"] = idx
            # Use high of current bar as starting highest_since_entry
            _order_state["entry_high"] = high_price if not np.isnan(high_price) else close_price
            return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

        # No entry -> no action
        return (float("nan"), SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # If in position, update trailing stop and check exit conditions
    else:
        # Update highest since entry
        if np.isnan(_order_state["entry_high"]):
            _order_state["entry_high"] = high_price
        else:
            # Use new highs
            _order_state["entry_high"] = max(_order_state["entry_high"], high_price)

        # Compute trailing stop level (highest_since_entry - trailing_mult * ATR)
        trailing_stop = np.nan
        if not np.isnan(atr_val):
            trailing_stop = _order_state["entry_high"] - float(trailing_mult) * atr_val

        # Exit conditions: MACD bearish cross OR price falls below trailing stop
        exit_by_macd = bool(cross_down)
        exit_by_trail = False
        if not np.isnan(trailing_stop):
            exit_by_trail = close_price < trailing_stop

        if exit_by_macd or exit_by_trail:
            # Close the long position by placing an opposite order
            _order_state["in_position"] = False
            _order_state["entry_idx"] = -1
            _order_state["entry_high"] = np.nan
            return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_SHORT)

        # No exit -> hold
        return (float("nan"), SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
