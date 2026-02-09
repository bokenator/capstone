"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- No numba usage
- compute_indicators is deterministic and uses only past data
- order_func keeps minimal per-run state on the function object and uses only past/current bars

"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def compute_indicators(
    ohlcv: Union[pd.DataFrame, pd.Series, np.ndarray],
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, Union[np.ndarray, pd.Series]]:
    """
    Compute indicators required by the strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    The function is flexible with input type: DataFrame (with columns like
    'open','high','low','close'), Series (treated as close), or numpy array
    (treated as close).

    All calculations use only historical data (no lookahead).
    """
    # Normalize input to pandas Series for ease of computation while preserving index
    if isinstance(ohlcv, pd.DataFrame):
        cols_map = {c.lower(): c for c in ohlcv.columns}

        def _get_col(name: str) -> pd.Series:
            if name in cols_map:
                return ohlcv[cols_map[name]].astype(float)
            # Fallbacks for common alternative names
            if name == "close":
                for alt in ("price", "adjclose", "adj_close"):
                    if alt in cols_map:
                        return ohlcv[cols_map[alt]].astype(float)
            # If not present, fall back to the last column
            return ohlcv.iloc[:, -1].astype(float)

        close_s = _get_col("close")
        high_s = _get_col("high") if "high" in cols_map else close_s
        low_s = _get_col("low") if "low" in cols_map else close_s

    elif isinstance(ohlcv, pd.Series):
        close_s = ohlcv.astype(float)
        high_s = close_s
        low_s = close_s

    else:
        # numpy array or list
        close_s = pd.Series(np.asarray(ohlcv, dtype=float))
        high_s = close_s
        low_s = close_s

    # Ensure a copy with a stable index
    close_s = pd.Series(close_s.values, index=close_s.index)
    high_s = pd.Series(high_s.values, index=close_s.index)
    low_s = pd.Series(low_s.values, index=close_s.index)

    # MACD (EMA-based)
    # Use adjust=False to avoid using future information
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_s = ema_fast - ema_slow
    signal_s = macd_s.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma_s = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Use simple rolling mean for ATR (min_periods=1 to avoid long NaN tail)
    atr_s = tr.rolling(window=atr_period, min_periods=1).mean()

    # Return numpy arrays for compatibility with vectorbt internals
    return {
        "close": close_s.values,
        "high": high_s.values,
        "macd": macd_s.values,
        "signal": signal_s.values,
        "atr": atr_s.values,
        "sma": sma_s.values,
    }


def order_func(
    c: Any,
    close: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    macd: Union[np.ndarray, pd.Series],
    signal: Union[np.ndarray, pd.Series],
    atr: Union[np.ndarray, pd.Series],
    sma: Union[np.ndarray, pd.Series],
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt.from_order_func (use_numba=False).

    Returns a tuple (size, size_type, direction) or (np.nan, ..) for no order.

    Strategy logic (long-only):
    - Entry when MACD crosses above Signal AND price > SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
    - Trailing stop updates to highest high since entry

    Implementation details:
    - Uses absolute "amount" orders (size_type=0) with positive size to buy and
      negative size to sell. This avoids ambiguity with enum mappings.
    - Stores minimal per-run state as attributes on the function object and
      re-initializes when a new run is detected.
    """
    # Convert inputs to numpy arrays for safe integer indexing
    close_a = np.asarray(close)
    high_a = np.asarray(high)
    macd_a = np.asarray(macd)
    signal_a = np.asarray(signal)
    atr_a = np.asarray(atr)
    sma_a = np.asarray(sma)

    # Current positional index provided by vectorbt context
    i = int(getattr(c, "i", 0))

    # Initialize per-run state when new data is detected
    last_id = getattr(order_func, "_last_close_id", None)
    last_len = getattr(order_func, "_last_len", None)
    current_id = id(close_a)
    current_len = len(close_a)

    if (last_id is None) or (last_id != current_id) or (last_len != current_len) or i == 0:
        order_func._in_position = False
        order_func._entry_high = -np.inf
        order_func._entry_index = None
        order_func._last_close_id = current_id
        order_func._last_len = current_len

    # Helper: safe previous index (use same bar when i == 0)
    prev_i = i - 1 if i > 0 else i

    # Read values safely (guarding against out-of-range or NaN)
    def _safe(arr: np.ndarray, idx: int) -> float:
        try:
            v = arr[idx]
            return float(v) if not np.isnan(v) else float("nan")
        except Exception:
            return float("nan")

    price = _safe(close_a, i)
    macd_cur = _safe(macd_a, i)
    signal_cur = _safe(signal_a, i)
    macd_prev = _safe(macd_a, prev_i)
    signal_prev = _safe(signal_a, prev_i)
    sma_val = _safe(sma_a, i)
    atr_val = _safe(atr_a, i)
    high_val = _safe(high_a, i)

    # Detect MACD crosses using only current and previous values (no future)
    cross_up = (macd_prev <= signal_prev) and (macd_cur > signal_cur)
    cross_down = (macd_prev >= signal_prev) and (macd_cur < signal_cur)

    # If not currently in position, check entry conditions
    if not getattr(order_func, "_in_position", False):
        if (
            not np.isnan(price)
            and not np.isnan(sma_val)
            and not np.isnan(macd_cur)
            and not np.isnan(signal_cur)
            and cross_up
            and price > sma_val
        ):
            # Enter: buy 1 unit (Amount semantics)
            size = 1.0
            size_type = 0  # Amount
            direction = 0  # Both (allow buying)

            order_func._in_position = True
            order_func._entry_index = i
            order_func._entry_high = high_val if not np.isnan(high_val) else price

            return (size, size_type, direction)

        return (float("nan"), 0, 0)

    # If in position, update highest_since_entry and evaluate exit rules
    if not np.isnan(high_val):
        order_func._entry_high = max(order_func._entry_high, high_val)

    if np.isnan(order_func._entry_high) or order_func._entry_high == -np.inf or np.isnan(atr_val):
        trailing_stop = float("-inf")
    else:
        trailing_stop = order_func._entry_high - (trailing_mult * atr_val)

    # Exit on MACD bearish cross or price crossing trailing stop
    if (
        (not np.isnan(macd_cur) and not np.isnan(signal_cur) and cross_down)
        or (not np.isnan(price) and price < trailing_stop)
    ):
        # Exit: sell 1 unit (Amount semantics, negative size to indicate sell)
        size = -1.0
        size_type = 0  # Amount
        direction = 0  # Both (allow selling)

        order_func._in_position = False
        order_func._entry_index = None
        order_func._entry_high = -np.inf

        return (size, size_type, direction)

    # No order
    return (float("nan"), 0, 0)
