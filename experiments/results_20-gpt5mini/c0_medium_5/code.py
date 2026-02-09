"""
MACD + ATR Trailing Stop Strategy

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- Pure Python implementation (no numba)
- Returns numpy arrays for indicators
- order_func returns tuple (size, size_type, direction) as expected by the provided wrapper

Strategy:
- Long entry when MACD crosses above Signal and price > SMA(50)
- Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

"""
from typing import Any, Dict, Tuple, Optional

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
    """Compute required indicators for the strategy.

    Args:
        ohlcv: DataFrame containing at least ['high', 'low', 'close'] columns.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for the trend SMA filter.
        atr_period: Period for ATR calculation.

    Returns:
        Dictionary with numpy arrays: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    # Validate input
    required_cols = ["close", "high", "low"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    # Ensure numeric series
    close_s = pd.Series(ohlcv["close"].astype(float)).reset_index(drop=True)
    high_s = pd.Series(ohlcv["high"].astype(float)).reset_index(drop=True)
    low_s = pd.Series(ohlcv["low"].astype(float)).reset_index(drop=True)

    # MACD: EMA(fast) - EMA(slow)
    # Use adjust=False for standard EMA
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=macd_signal, adjust=False).mean().values

    # SMA (trend filter)
    # Use min_periods=sma_period so SMA is NaN until enough data (common practice)
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean().values

    # ATR: True Range then simple moving average
    prev_close = close_s.shift(1)
    tr_1 = high_s - low_s
    tr_2 = (high_s - prev_close).abs()
    tr_3 = (low_s - prev_close).abs()
    tr = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean().values

    return {
        "close": close_s.values,
        "high": high_s.values,
        "macd": np.asarray(macd_line, dtype=float),
        "signal": np.asarray(signal_line, dtype=float),
        "atr": np.asarray(atr, dtype=float),
        "sma": np.asarray(sma, dtype=float),
    }


def _get_attr_or_call(obj: Any, name: str) -> Optional[Any]:
    """Helper: get attribute or call method if callable. Returns None if not present or fails."""
    if not hasattr(obj, name):
        return None
    val = getattr(obj, name)
    try:
        return val() if callable(val) else val
    except Exception:
        try:
            # Some context objects may want no-arg call but raise different errors; swallow
            return None
        except Exception:
            return None


def _get_current_position_size(c: Any) -> float:
    """Robustly extract current position size from the context. Returns 0.0 if unknown."""
    # Try common attribute names
    candidates = [
        "size",
        "pos",
        "position",
        "position_size",
        "current_size",
        "current_pos",
    ]
    for name in candidates:
        val = _get_attr_or_call(c, name)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue

    # Try common getter methods
    methods = [
        "get_size",
        "get_pos",
        "get_position",
        "get_current_size",
        "get_current_pos",
    ]
    for name in methods:
        val = _get_attr_or_call(c, name)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue

    # Fallback: use is_open / is_long boolean
    try:
        is_open = _get_attr_or_call(c, "is_open")
        if is_open:
            # If open but we couldn't read size, return a positive proxy (1.0)
            return 1.0
    except Exception:
        pass

    try:
        is_long = _get_attr_or_call(c, "is_long")
        if is_long:
            return 1.0
    except Exception:
        pass

    return 0.0


def _get_last_entry_index(c: Any) -> Optional[int]:
    """Try to retrieve the last entry index for the currently open position.

    Returns integer index (0-based) or None if unavailable.
    """
    # Candidate method/attribute names that may exist on the context
    candidates = [
        "get_last_entry_i",
        "get_last_entry_idx",
        "get_last_entry_index",
        "get_last_entry",
        "last_entry_i",
        "entry_i",
        "entry_idx",
        "entry_index",
    ]
    for name in candidates:
        val = _get_attr_or_call(c, name)
        if val is None:
            continue
        # If the returned value is an index-like thing, try to convert
        try:
            if isinstance(val, (list, tuple, np.ndarray, pd.Index)):
                if len(val) == 0:
                    continue
                return int(val[-1])
            return int(val)
        except Exception:
            continue
    return None


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
    """Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    Returns a tuple (size, size_type, direction):
      - size: float (np.nan to indicate no order)
      - size_type: int (0=Amount, 1=Percent, etc.); we use 1 (Percent) for target sizing
      - direction: int (1=Long)

    Logic:
      - Entry: when MACD crosses above Signal AND price > SMA(50) -> set target 100% long
      - Exit: when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR) -> set target 0%

    Notes:
      - This implementation is defensive: it tolerates missing context attributes and NaNs.
    """
    # Defensive checks
    try:
        i = int(c.i)
    except Exception:
        # If we cannot determine the index, do nothing
        return (float("nan"), 0, 0)

    n = len(close)
    if i < 1 or i >= n:
        return (float("nan"), 0, 0)

    # Current position size (proxy). If > 0 -> long position open
    cur_size = _get_current_position_size(c)
    in_long = cur_size > 0

    # MACD crossover detection (ensure no NaNs)
    macd_prev = macd[i - 1]
    macd_curr = macd[i]
    sig_prev = signal[i - 1]
    sig_curr = signal[i]

    macd_cross_up = False
    macd_cross_down = False
    try:
        if not (np.isnan(macd_prev) or np.isnan(macd_curr) or np.isnan(sig_prev) or np.isnan(sig_curr)):
            macd_cross_up = (macd_prev <= sig_prev) and (macd_curr > sig_curr)
            macd_cross_down = (macd_prev >= sig_prev) and (macd_curr < sig_curr)
    except Exception:
        macd_cross_up = False
        macd_cross_down = False

    # Price vs SMA
    price = float(close[i])
    sma_val = float(sma[i]) if not np.isnan(sma[i]) else float("nan")
    price_above_sma = False
    if not np.isnan(sma_val):
        price_above_sma = price > sma_val

    # ENTRY: long-only
    if (not in_long) and macd_cross_up and price_above_sma:
        # Use SizeType=Percent (1) and Direction=Long (1) to set target 100% long
        return (1.0, 1, 1)

    # EXIT conditions (only if currently long)
    if in_long:
        # MACD cross down
        if macd_cross_down:
            # Set target percent to 0 to exit
            return (0.0, 1, 1)

        # Trailing stop based on highest high since entry
        last_entry_idx = _get_last_entry_index(c)
        highest_since_entry = None
        try:
            if last_entry_idx is not None and 0 <= last_entry_idx <= i:
                highest_since_entry = float(np.nanmax(high[last_entry_idx : i + 1]))
            else:
                # If we cannot determine entry, be conservative: consider highest since position likely opened
                # but to avoid false triggers we default to current price as highest (disables trailing stop)
                highest_since_entry = float(price)
        except Exception:
            highest_since_entry = float(price)

        atr_val = float(atr[i]) if not np.isnan(atr[i]) else float("nan")
        if not np.isnan(atr_val):
            trailing_level = highest_since_entry - trailing_mult * atr_val
            if price < trailing_level:
                return (0.0, 1, 1)

    # No action
    return (float("nan"), 0, 0)
