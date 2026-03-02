"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
  -> dict of numpy arrays: macd, signal, atr, sma, close, high

- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)
  -> returns tuple (size, size_type, direction)

Notes:
- No numba usage.
- Keeps lightweight Python state on the order_func function object to track position and highest price since entry.
- Uses Amount sizing (size_type=1) and positive size for buy, negative size for sell.

This file will be passed to a backtest runner that wraps the returned tuple into vectorbt orders.
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
    Compute MACD, Signal, SMA and ATR arrays from OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (only high/low/close required)
        macd_fast: fast EMA period for MACD
        macd_slow: slow EMA period for MACD
        macd_signal: signal line EMA period
        sma_period: period for trend SMA
        atr_period: period for ATR

    Returns:
        Dictionary containing numpy arrays for keys: macd, signal, atr, sma, close, high
    """
    # Validate input columns
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD: EMA difference and signal line (Wilder/EMA style)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_series = ema_fast - ema_slow
    signal_series = macd_series.ewm(span=macd_signal, adjust=False).mean()

    # SMA for trend filter
    sma_series = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR: True Range then Wilder smoothing (EMA with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # Ensure arrays are numpy floats
    out = {
        "macd": macd_series.fillna(0.0).to_numpy(dtype=float),
        "signal": signal_series.fillna(0.0).to_numpy(dtype=float),
        "atr": atr_series.fillna(method="bfill").fillna(0.0).to_numpy(dtype=float),
        "sma": sma_series.fillna(method="bfill").to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        "high": high.to_numpy(dtype=float),
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
    Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    Maintains simple Python state on the function object to track whether we are in a long
    position, the entry index and the highest price seen since entry (for ATR trailing stop).

    Return value is a tuple: (size, size_type, direction)
    - size: positive for buy, negative for sell. Returning np.nan for size means no order.
    - size_type: integer code for size type. We use `1` to represent Amount (matches typical vectorbt enum).
    - direction: integer code for direction. We use `1` to represent a long-direction order.

    Args:
        c: context object provided by vectorbt (has attribute `i` for current integer bar index)
        close: numpy array of close prices
        high: numpy array of high prices
        macd: numpy array of MACD line
        signal: numpy array of MACD signal line
        atr: numpy array of ATR values
        sma: numpy array of SMA values
        trailing_mult: multiplier for ATR to compute trailing stop (e.g., 2.0)

    Notes:
        - This function intentionally avoids numba and low-level vectorbt.nb calls.
        - Uses a fixed amount-based sizing (1 unit) for simplicity.
    """
    # Initialize lightweight persistent state on the function object
    if not hasattr(order_func, "_state"):
        # _state stores a single-entry state dict for this single-asset strategy
        order_func._state = {
            "in_position": False,
            "entry_idx": -1,
            "highest": -np.inf,
            "position_size": 0.0,
        }

    state = order_func._state

    # Get current index
    if not hasattr(c, "i"):
        # If context is missing index, do nothing
        return (np.nan, 1, 1)

    i = int(c.i)

    # Safety checks: array bounds
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 1, 1)

    # Pull current bar values
    price = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan

    # For crosses we need previous values
    if i == 0:
        # No previous bar to compare
        prev_macd = np.nan
        prev_signal = np.nan
    else:
        prev_macd = float(macd[i - 1])
        prev_signal = float(signal[i - 1])

    macd_i = float(macd[i])
    signal_i = float(signal[i])
    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # If essential indicators are NaN, skip
    if np.isnan(price) or np.isnan(macd_i) or np.isnan(signal_i) or np.isnan(atr_i) or np.isnan(sma_i):
        return (np.nan, 1, 1)

    # Entry logic: MACD crosses above signal and price above SMA
    if not state["in_position"]:
        crossed_up = (not np.isnan(prev_macd) and not np.isnan(prev_signal)) and (prev_macd <= prev_signal and macd_i > signal_i)
        if crossed_up and price > sma_i:
            # Enter long: use 1 unit amount sizing
            size = 1.0
            state["in_position"] = True
            state["entry_idx"] = i
            state["highest"] = high_i if not np.isnan(high_i) else price
            state["position_size"] = float(size)
            # Return (size, size_type, direction) -> positive size for buy
            return (float(size), 1, 1)
        else:
            return (np.nan, 1, 1)

    # If already in a position, check exit conditions
    # Update highest price seen since entry
    if not np.isnan(high_i):
        state["highest"] = max(state["highest"], high_i)

    # Compute trailing stop level
    trailing_stop = state["highest"] - trailing_mult * atr_i

    crossed_down = (not np.isnan(prev_macd) and not np.isnan(prev_signal)) and (prev_macd >= prev_signal and macd_i < signal_i)
    fell_below_trail = price < trailing_stop if not np.isnan(trailing_stop) else False

    if crossed_down or fell_below_trail:
        # Exit: sell the full position by returning negative amount equal to stored position_size
        size = -abs(state["position_size"]) if state["position_size"] != 0 else np.nan
        # reset state
        state["in_position"] = False
        state["entry_idx"] = -1
        state["highest"] = -np.inf
        state["position_size"] = 0.0
        if np.isnan(size):
            return (np.nan, 1, 1)
        return (float(size), 1, 1)

    # Otherwise hold
    return (np.nan, 1, 1)
