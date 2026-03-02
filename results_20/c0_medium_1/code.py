"""
MACD + ATR Trailing Stop strategy for vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> dict[str, np.ndarray]
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple[float, int, int]

Notes:
- No numba usage
- Does not import vbt.portfolio.enums
- order_func returns tuples (size, size_type, direction). The backtest wrapper will convert them to orders.
  We use SizeType=2 (TargetPercent) and Direction=1 (Long-only) as integer constants so that
  size=1.0 means target 100% exposure and size=0.0 means close (0%).

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
    """Compute indicators required by the strategy.

    Args:
        ohlcv: DataFrame with at least ['open','high','low','close'] columns.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for the trend SMA filter.
        atr_period: Period for ATR.

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma' containing numpy arrays.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    open_ = ohlcv["open"].astype(float)

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd, signal)
    # Use pandas ewm with adjust=False for standard EMA behavior.
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # ATR: True Range and Wilder's smoothing via ewm with alpha=1/period
    prev_close = close.shift(1)
    high_low = high - low
    high_prev_close = (high - prev_close).abs()
    low_prev_close = (low - prev_close).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # Wilder's moving average: alpha = 1/atr_period, adjust=False
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")
    atr = true_range.ewm(alpha=1.0 / float(atr_period), adjust=False, min_periods=1).mean()

    # SMA for trend filter
    if sma_period <= 0:
        raise ValueError("sma_period must be > 0")
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # Return numpy arrays; ensure dtype=float64
    return {
        "close": close.values.astype(float),
        "high": high.values.astype(float),
        "macd": macd.values.astype(float),
        "signal": signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
    }


def _safe_get_attr_any(obj: Any, candidates: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first existing attribute from candidates or default."""
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


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
    """Order function for vectorbt.from_order_func (use_numba=False).

    Returns a tuple (size, size_type, direction).
    - We use size_type = 2 (TargetPercent) and direction = 1 (Long-only).
      This means size=1.0 -> target 100% long exposure; size=0.0 -> close.

    Logic:
    - Entry (when flat): MACD crosses above signal AND price > 50-SMA
    - Exit (when long): MACD crosses below signal OR price < (highest_since_entry - trailing_mult*ATR)

    Notes:
    - We attempt to be robust to different context attribute names by checking several common
      attribute candidates. If essential information is missing, the function will be conservative
      and not emit orders.
    """
    # Constants for returned tuple
    # Use integer literals instead of importing vectorbt enums to comply with constraints.
    SIZE_TYPE_TARGET_PERCENT = 2  # Target percent (set allocation to a target fraction)
    DIRECTION_LONG = 1  # Long-only direction

    # Get current index
    i = _safe_get_attr_any(c, ("i", "index", "idx"), None)
    try:
        i = int(i)
    except Exception:
        # If we can't determine current index, be safe and do nothing
        return (float("nan"), 0, 0)

    n = len(close)
    if i < 0 or i >= n:
        return (float("nan"), 0, 0)

    # Helper to get prior value
    def prev_arr(arr: np.ndarray, idx: int) -> float:
        if idx - 1 < 0:
            return np.nan
        return float(arr[idx - 1])

    # Read indicator values at current bar
    curr_macd = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    curr_signal = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    prev_macd = prev_arr(macd, i)
    prev_signal = prev_arr(signal, i)
    curr_close = float(close[i]) if not np.isnan(close[i]) else np.nan
    curr_sma = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    curr_atr = float(atr[i]) if not np.isnan(atr[i]) else np.nan

    # Determine current position size (if any). Several context attribute names are possible.
    pos = _safe_get_attr_any(c, ("pos", "position", "position_size", "size", "cur_pos"), 0)
    try:
        pos = float(pos)
    except Exception:
        # If pos can't be parsed, assume flat
        pos = 0.0

    # Detect MACD cross
    macd_cross_up = False
    macd_cross_down = False
    if not np.isnan(prev_macd) and not np.isnan(prev_signal) and not np.isnan(curr_macd) and not np.isnan(curr_signal):
        macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
        macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

    # Price above SMA (trend filter)
    price_above_sma = False
    if not np.isnan(curr_close) and not np.isnan(curr_sma):
        price_above_sma = curr_close > curr_sma

    # ENTRY: only when currently flat or not long
    is_flat = pos == 0 or pos == 0.0
    if is_flat and macd_cross_up and price_above_sma:
        # Enter full allocation (100% target percent)
        return (1.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_LONG)

    # EXIT: when currently long
    is_long = pos > 0
    if is_long:
        # Attempt to obtain the index of the last entry to compute highest since entry.
        entry_idx = _safe_get_attr_any(
            c,
            (
                "entry_idx",
                "entry_i",
                "last_entry_idx",
                "last_entry_i",
                "entry_index",
            ),
            None,
        )
        # If an entries list is provided, try to take its last element
        if entry_idx is None and hasattr(c, "entries"):
            try:
                entries = getattr(c, "entries")
                if entries is not None and len(entries) > 0:
                    entry_idx = int(entries[-1])
            except Exception:
                entry_idx = None

        # Fallback: if we don't know entry idx, use the earliest possible (0)
        if entry_idx is None:
            entry_idx = 0
        try:
            entry_idx = int(entry_idx)
        except Exception:
            entry_idx = 0

        # Bound entry_idx
        entry_idx = max(0, min(entry_idx, i))

        # Compute highest high since entry (inclusive)
        try:
            if entry_idx <= i:
                highest_since_entry = float(np.nanmax(high[entry_idx : i + 1]))
            else:
                highest_since_entry = float(np.nanmax(high[: i + 1]))
        except Exception:
            highest_since_entry = np.nan

        # Compute trailing stop price
        stop_price = np.nan
        if not np.isnan(highest_since_entry) and not np.isnan(curr_atr):
            stop_price = highest_since_entry - float(trailing_mult) * curr_atr

        # Exit if MACD crossed down or price fell below trailing stop
        price_below_stop = False
        if not np.isnan(stop_price) and not np.isnan(curr_close):
            price_below_stop = curr_close < stop_price

        if macd_cross_down or price_below_stop:
            # Close position (target percent 0)
            return (0.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_LONG)

    # Otherwise, no order
    return (float("nan"), 0, 0)
