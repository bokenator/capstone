import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


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

    Returns a dict with numpy arrays for keys:
      - 'close': close prices
      - 'high': high prices
      - 'macd': MACD line (fast EMA - slow EMA)
      - 'signal': MACD signal line
      - 'atr': Average True Range
      - 'sma': Simple Moving Average (trend filter)

    All arrays are the same length as ohlcv and contain np.nan where not
    computable (warmup).
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ("open", "high", "low", "close"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv missing required column: {col}")

    close = ohlcv["close"].astype(float).copy()
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd, macd_signal)
    # Use pandas ewm with adjust=False (typical for TA)
    fast_ema = close.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # ATR: True Range then rolling mean (simple) over atr_period
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # Convert to numpy arrays
    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr.values,
        "sma": sma.values,
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
    Order function for vectorbt.from_order_func (use_numba=False).

    Returns a tuple: (size, size_type, direction)
      - Enter: set target percent to 1.0 (100% of equity)
      - Exit: set target percent to 0.0 (close position)
      - No action: return (np.nan, Amount, Both)

    We avoid direct imports of vectorbt enums here. Based on vectorbt's
    internal enums the following integer mapping is assumed:
      - SizeType.Amount = 0 (used for the no-op np.nan return)
      - SizeType.TargetPercent = 2 (used to set full/zero target)
      - Direction.Both = 0
      - Direction.Long = 1

    The function is defensive and handles NaNs / warmup periods.
    """
    # Defensive access to current index
    i = int(getattr(c, "i", 0))

    # Helper to safely test a crossover up (macd crosses above signal)
    def cross_up(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = arr1[idx - 1], arr2[idx - 1]
        a_cur, b_cur = arr1[idx], arr2[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev <= b_prev) and (a_cur > b_cur)

    def cross_down(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = arr1[idx - 1], arr2[idx - 1]
        a_cur, b_cur = arr1[idx], arr2[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_cur) or np.isnan(b_cur):
            return False
        return (a_prev >= b_prev) and (a_cur < b_cur)

    # Safe retrieval of price and indicator values
    price = float(close[i]) if i < len(close) else np.nan
    sma_val = float(sma[i]) if i < len(sma) else np.nan
    atr_val = float(atr[i]) if i < len(atr) else np.nan

    # Determine whether we are currently in a position
    in_position = False
    if hasattr(c, "is_open"):
        try:
            in_position = bool(c.is_open)
        except Exception:
            in_position = False
    elif hasattr(c, "position"):
        try:
            in_position = bool(c.position)
        except Exception:
            in_position = False

    # Entry condition: MACD cross up AND price above SMA
    enter_signal = cross_up(macd, signal, i) and (not np.isnan(sma_val)) and (price > sma_val)

    # Exit conditions: MACD cross down OR price drops below trailing stop
    macd_cross_down = cross_down(macd, signal, i)

    # Compute highest price since entry for trailing stop
    highest_since_entry = np.nan
    # Prefer using context helper if available
    if hasattr(c, "max_since_entry"):
        try:
            # c.max_since_entry expects the full array and returns scalar
            highest_since_entry = float(c.max_since_entry(high))
        except Exception:
            highest_since_entry = np.nan
    else:
        # Fallback: try to use entry index if available
        entry_idx = None
        for attr in ("entry_idx", "entry_i", "entry_index"):
            if hasattr(c, attr):
                try:
                    entry_idx = int(getattr(c, attr))
                    break
                except Exception:
                    entry_idx = None
        if entry_idx is not None and entry_idx >= 0:
            try:
                highest_since_entry = float(np.nanmax(high[entry_idx : i + 1]))
            except Exception:
                highest_since_entry = np.nan
        else:
            # As a last resort use current high
            try:
                highest_since_entry = float(high[i])
            except Exception:
                highest_since_entry = np.nan

    trailing_stop_price = (
        highest_since_entry - trailing_mult * atr_val
        if (not np.isnan(highest_since_entry) and not np.isnan(atr_val))
        else np.nan
    )

    # If not in position and entry signal -> enter (target 100% long)
    # Use TargetPercent=2, Direction.Long=1
    if (not in_position) and enter_signal:
        return (1.0, 2, 1)

    # If in position, check exits
    if in_position:
        # Exit on MACD cross down
        if macd_cross_down:
            return (0.0, 2, 1)

        # Exit on trailing stop breach
        if (not np.isnan(trailing_stop_price)) and (price < trailing_stop_price):
            return (0.0, 2, 1)

    # No action: return NaN size
    # Use SizeType.Amount=0 and Direction.Both=0 for the no-op wrapper case
    return (np.nan, 0, 0)
