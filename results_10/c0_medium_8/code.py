# coding: utf-8
"""
MACD + ATR Trailing Stop strategy for vectorbt from_order_func (no numba).

Exports:
- compute_indicators(ohlcv, macd_fast, macd_slow, macd_signal, sma_period, atr_period) -> dict[str, np.ndarray]
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple[float, int, int]

Notes:
- Uses EMA for MACD and Wilder-like EMA for ATR (EWMA with alpha=1/atr_period).
- Long-only. Entry: MACD crosses above signal AND price > sma. Exit: MACD crosses below OR price < (highest_since_entry - trailing_mult * ATR).
- Trailing stop uses the highest high since the last entry index provided by the order context (c.entry_i or variations).
- Does not use numba.

Author: Assistant
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

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least high, low, close required).
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for trend filter SMA.
        atr_period: Period for ATR.

    Returns:
        Dict with numpy arrays: close, high, macd, signal, atr, sma
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ("close", "high", "low"):
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # MACD
    # Use EMA (pandas ewm with adjust=False)
    fast_ema = close_s.ewm(span=int(macd_fast), adjust=False).mean()
    slow_ema = close_s.ewm(span=int(macd_slow), adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=int(macd_signal), adjust=False).mean()

    # SMA trend filter
    # Use min_periods=1 to avoid long NaN run; strategy will naturally ignore early signals
    sma = close_s.rolling(window=int(sma_period), min_periods=1).mean()

    # ATR (True Range then Wilder-like EMA)
    prev_close = close_s.shift(1)
    tr_1 = high_s - low_s
    tr_2 = (high_s - prev_close).abs()
    tr_3 = (low_s - prev_close).abs()
    tr = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    # Wilder smoothing approx via ewm with alpha=1/atr_period and adjust=False
    atr = tr.ewm(alpha=1.0 / float(max(1, int(atr_period))), adjust=False).mean()

    # Convert to numpy arrays
    out = {
        "close": close_s.values.astype(float),
        "high": high_s.values.astype(float),
        "macd": macd.values.astype(float),
        "signal": signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
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
    Order function for vectorbt.Portfolio.from_order_func (no numba).

    Returns tuple (size, size_type, direction) or a no-op tuple with size = np.nan.

    Logic:
    - If not in position:
        - Enter long when MACD crosses above signal and price > sma -> buy 1 unit (Amount)
    - If in position:
        - Exit when MACD crosses below signal
        - Exit when price < (highest_since_entry - trailing_mult * ATR)

    Notes about the order tuple values:
    - We return size as a positive number to buy and a negative number to sell when direction allows both sides.
    - We use SizeType = 0 (Amount) and Direction = 0 (Both) for the orders. The backtest runner's wrapper
      will convert the tuple to vectorbt order objects. This mapping (Amount=0, Both=0) is compatible with
      vectorbt's internal enums in typical versions.

    Args:
        c: order function context provided by vectorbt (has attributes like i, in_position, entry_i)
        close, high, macd, signal, atr, sma: numpy arrays of indicators
        trailing_mult: multiplier for ATR in trailing stop (e.g., 2.0)

    Returns:
        (size, size_type, direction) tuple. Use np.nan for size to indicate no action.
    """
    i = int(getattr(c, "i", 0))

    # Helper: safe access of previous values
    def prev_idx(idx: int) -> int:
        return max(0, idx - 1)

    # Determine whether currently in a position.
    # Vectorbt context usually exposes `in_position`. Try common alternatives for robustness.
    in_pos = bool(
        getattr(c, "in_position", False)
        or getattr(c, "in_pos", False)
        or getattr(c, "is_open", False)
    )

    # Basic no-op return (size is np.nan means no order)
    no_op: Tuple[float, int, int] = (np.nan, 0, 0)

    # Make sure indices are within array bounds
    n = len(close)
    if i < 0 or i >= n:
        return no_op

    # Ensure we have finite indicator values at current bar
    if not np.isfinite(macd[i]) or not np.isfinite(signal[i]) or not np.isfinite(sma[i]) or not np.isfinite(atr[i]):
        return no_op

    # Detect cross up and down with previous bar
    pi = prev_idx(i)
    cross_up = False
    cross_down = False
    if i > 0 and np.isfinite(macd[pi]) and np.isfinite(signal[pi]):
        cross_up = (macd[pi] <= signal[pi]) and (macd[i] > signal[i])
        cross_down = (macd[pi] >= signal[pi]) and (macd[i] < signal[i])

    # Size / type / direction convention used here:
    # size_type = 0 -> Amount, direction = 0 -> Both. Use positive size to buy, negative to sell.
    buy_size = 1.0
    sell_size = -1.0
    size_type_amount = 0
    direction_both = 0

    # Entry logic
    if not in_pos:
        # Require MACD cross above AND price above SMA
        if cross_up and close[i] > sma[i]:
            return (float(buy_size), int(size_type_amount), int(direction_both))
        return no_op

    # When in position: compute highest price since entry and apply exit rules
    # Try to get the last entry index from context. Vectorbt typically provides c.entry_i.
    last_entry_i = None
    for attr in ("entry_i", "entry_idx", "entry_index", "entry_index_i"):
        last_entry_i = getattr(c, attr, None)
        if last_entry_i is not None:
            break

    # If we couldn't find last_entry_i, fallback to 0 to compute highest conservatively
    try:
        last_entry_i = int(last_entry_i) if last_entry_i is not None else 0
    except Exception:
        last_entry_i = 0

    last_entry_i = max(0, min(i, last_entry_i))

    # Highest high since entry (inclusive)
    try:
        highest_since_entry = float(np.nanmax(high[last_entry_i : i + 1]))
    except Exception:
        highest_since_entry = float(high[i]) if np.isfinite(high[i]) else np.nan

    # Exit on MACD cross down
    if cross_down:
        return (float(sell_size), int(size_type_amount), int(direction_both))

    # Exit on ATR-based trailing stop
    if np.isfinite(highest_since_entry) and np.isfinite(atr[i]):
        trailing_stop = highest_since_entry - float(trailing_mult) * float(atr[i])
        if np.isfinite(trailing_stop) and close[i] < trailing_stop:
            return (float(sell_size), int(size_type_amount), int(direction_both))

    return no_op
