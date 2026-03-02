# Generated strategy combining MACD crossover entries with ATR-based trailing stops
# Requirements:
# - MACD: fast=12, slow=26, signal=9
# - Trend filter: 50-period SMA
# - ATR period: 14
# - Trailing stop: 2.0 * ATR from highest price since entry
# - Long-only, single asset
#
# Two exported functions:
# - compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> dict[str, np.ndarray]
# - order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple[float, int, int]

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
    Compute indicators required for the strategy.

    Returns a dict with numpy arrays for keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (or at least high, low, close).
        macd_fast, macd_slow, macd_signal: MACD parameters.
        sma_period: Period for the trend SMA.
        atr_period: Period for ATR.

    Returns:
        Dict[str, np.ndarray]
    """
    # Validate input columns
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    # Ensure numeric floats and copy to avoid modifying original
    close_s = pd.Series(ohlcv["close"].astype(float).values)
    high_s = pd.Series(ohlcv["high"].astype(float).values)
    low_s = pd.Series(ohlcv["low"].astype(float).values)

    # MACD (EMA-based)
    # Use pandas ewm with span so that typical MACD behaviour is followed
    fast_ema = close_s.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close_s.rolling(window=sma_period, min_periods=1).mean()

    # True Range (TR)
    prev_close = close_s.shift(1)
    tr_0 = high_s - low_s
    tr_1 = (high_s - prev_close).abs()
    tr_2 = (low_s - prev_close).abs()
    tr = pd.concat([tr_0, tr_1, tr_2], axis=1).max(axis=1)

    # ATR using Wilder's smoothing (ewm with alpha=1/period, adjust=False)
    # Convert atr_period to float to avoid integer division issues in older Pythons
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # Convert to numpy arrays and return
    return {
        "macd": macd_line.to_numpy(dtype=float),
        "signal": signal_line.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "sma": sma.to_numpy(dtype=float),
        "close": close_s.to_numpy(dtype=float),
        "high": high_s.to_numpy(dtype=float),
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
    Order function to be used with vectorbt.Portfolio.from_order_func (use_numba=False).

    Returns a tuple: (size, size_type, direction)
      - size: positive to buy, negative to sell, np.nan for no action
      - size_type: integer code (0 = Amount)
      - direction: integer code (0 = Both)

    Important: We avoid importing vectorbt enums here. We use convention size_type=0 (Amount), direction=0 (Both).

    Logic:
      - Entry (when not in position): MACD crosses above Signal AND close > SMA
      - Exit (when in position): MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    The function attempts to read the current position size and entry index from the context object 'c'. It is defensive
    about attribute names to be compatible with different vectorbt versions.
    """
    # Constants: do not import enums to remain compatible with runner constraints
    SIZE_TYPE_AMOUNT = 0  # SizeType.Amount
    DIRECTION_BOTH = 0  # Direction.Both

    # Current bar index
    i = int(getattr(c, "i", 0))

    # Safety: if index out of bounds for provided arrays, do nothing
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Helper to get current position size (float). Default 0.0 (no position)
    pos_size = 0.0
    entry_idx = None
    try:
        position_obj = getattr(c, "position", None)
        if position_obj is None:
            # Fallbacks: older interfaces may expose different attributes
            pos_size = float(getattr(c, "position_size", getattr(c, "size", 0.0)))
        else:
            # position_obj might be a numeric or an object with attributes
            if isinstance(position_obj, (int, float, np.integer, np.floating)):
                pos_size = float(position_obj)
            else:
                # Try common attribute names
                # size
                if hasattr(position_obj, "size"):
                    try:
                        pos_size = float(getattr(position_obj, "size"))
                    except Exception:
                        # Some objects may expose size as a numpy scalar
                        pos_size = float(np.asarray(getattr(position_obj, "size")))
                else:
                    # Fallbacks
                    pos_size = float(getattr(position_obj, "current_size", getattr(position_obj, "size_", 0.0)))

                # entry index: try several common names
                entry_idx = getattr(position_obj, "entry_idx", None)
                if entry_idx is None:
                    entry_idx = getattr(position_obj, "entry_i", None)
                if entry_idx is None:
                    entry_idx = getattr(position_obj, "entry_index", None)
    except Exception:
        # Be defensive; do not raise inside order_func
        pos_size = 0.0
        entry_idx = None

    # Convert pos_size to float (ensure numpy scalar ok)
    try:
        pos_size = float(pos_size)
    except Exception:
        pos_size = 0.0

    # Convert entry_idx to int if present
    if entry_idx is not None:
        try:
            entry_idx = int(entry_idx)
        except Exception:
            entry_idx = None

    # Prevent acting on NaN indicator values
    if np.isnan(macd[i]) or np.isnan(signal[i]) or np.isnan(close[i]) or np.isnan(sma[i]):
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Helper to detect crosses safely (requires previous bar)
    def cross_up(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx], arr_b[idx]
        a1, b1 = arr_a[idx - 1], arr_b[idx - 1]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 > b0) and (a1 <= b1)

    def cross_down(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx], arr_b[idx]
        a1, b1 = arr_a[idx - 1], arr_b[idx - 1]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 < b0) and (a1 >= b1)

    # ENTRY: if not currently in a long position
    in_long = pos_size > 0

    entry_signal = cross_up(macd, signal, i) and (close[i] > sma[i])

    if (not in_long) and entry_signal:
        # Enter a single unit (amount). Use direction=Both so sign controls buy/sell.
        return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # EXIT: if currently in long position, check conditions
    if in_long:
        # 1) MACD cross below signal
        if cross_down(macd, signal, i):
            # Close entire position: return negative of current size
            # If pos_size is zero for some reason, close one unit as fallback
            size = -pos_size if pos_size != 0.0 else -1.0
            return (float(size), SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

        # 2) Trailing stop based on highest price since entry
        # Need atr value available
        if not np.isnan(atr[i]) and entry_idx is not None and entry_idx <= i:
            # Compute highest high since entry (inclusive)
            try:
                highest = float(np.nanmax(high[entry_idx : i + 1]))
            except Exception:
                highest = float(np.nanmax(high[: i + 1]))

            stop_level = highest - float(trailing_mult) * float(atr[i])
            if close[i] < stop_level:
                size = -pos_size if pos_size != 0.0 else -1.0
                return (float(size), SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Default: no action
    return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
