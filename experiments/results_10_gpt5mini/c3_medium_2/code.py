"""
MACD + ATR Trailing Stop strategy implementation for vectorbt backtests.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> dict[str, np.ndarray]
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple

Notes:
- No numba usage.
- Designed to be deterministic and avoid lookahead: only uses c.i and past data (i and i-1).
- order_func keeps minimal per-run state as function attributes and resets at the start of each run (c.i == 0).

Strategy logic:
- Long entries when MACD crosses above Signal AND price > SMA(50).
- Exit when MACD crosses below Signal OR price falls below (highest_since_entry - trailing_mult * ATR).
- Trailing stop uses highest high since entry.

"""
from typing import Any, Dict, Optional, Tuple

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

    Accepts either a DataFrame with OHLCV columns or a pandas Series / 1D numpy array of closes.

    Returns a dict with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    All returned arrays are 1D numpy arrays aligned with the input index.
    """
    # Normalize input to DataFrame with open/high/low/close
    if isinstance(ohlcv, pd.Series) or isinstance(ohlcv, np.ndarray):
        close_series = pd.Series(ohlcv)
        df = pd.DataFrame({
            "open": close_series,
            "high": close_series,
            "low": close_series,
            "close": close_series,
        })
    elif isinstance(ohlcv, pd.DataFrame):
        df = ohlcv.copy()
        # Ensure required columns exist
        if "close" not in df.columns:
            raise ValueError("ohlcv DataFrame must contain 'close' column")
        for col in ["open", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"]
    else:
        # Fallback: try to convert
        close_series = pd.Series(ohlcv)
        df = pd.DataFrame({
            "open": close_series,
            "high": close_series,
            "low": close_series,
            "close": close_series,
        })

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # MACD (EMA fast - EMA slow) and Signal line (EMA of MACD)
    # Use adjust=False for standard EMA behavior
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # Simple moving average as trend filter
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR: True Range and rolling mean (simple ATR)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    # Convert to numpy arrays
    return {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
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
    Order function for vectorbt's from_order_func.

    Parameters
    - c: context object provided by vectorbt (provides c.i for current index)
    - close, high, macd, signal, atr, sma: numpy arrays aligned with the data
    - trailing_mult: multiplier for ATR to compute trailing stop (e.g., 2.0)

    Returns a tuple (size, size_type, direction) where:
    - size: float (use np.nan for no order)
    - size_type: int (0 => Amount assumed)
    - direction: int (1 => Long entry, 2 => Short/Exit)

    Implementation notes:
    - The function maintains minimal state on itself as attributes:
      - position_open: bool
      - entry_index: Optional[int]
      - highest_since_entry: float
    - State is reset at the start of each run (when c.i == 0).

    No lookahead is used: only c.i and c.i-1 are accessed.
    """
    # Ensure arrays are numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    # Current index
    i = int(getattr(c, "i", 0))

    # Initialize / reset per-run state when starting at index 0
    if i == 0:
        order_func.position_open = False
        order_func.entry_index = None
        order_func.highest_since_entry = -np.inf
        order_func.entry_price = np.nan

    # Read persistent state
    position_open: bool = bool(getattr(order_func, "position_open", False))
    entry_index: Optional[int] = getattr(order_func, "entry_index", None)
    highest_since_entry: float = float(getattr(order_func, "highest_since_entry", -np.inf))

    # Helper to emit no-op (no order)
    def no_order() -> Tuple[float, int, int]:
        # (np.nan size => wrapper will convert to no order)
        return (np.nan, 0, 0)

    # Bounds check
    n = len(close)
    if i < 0 or i >= n:
        return no_order()

    # Current and previous indicator values (use current if previous not available)
    curr_macd = macd[i]
    curr_signal = signal[i]
    prev_macd = macd[i - 1] if i > 0 else curr_macd
    prev_signal = signal[i - 1] if i > 0 else curr_signal

    # Ensure numeric (handle NaN safely)
    def is_valid(x: float) -> bool:
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    # When not in position, check for entry
    if not position_open:
        # Require valid MACD & Signal & SMA for entry
        if is_valid(curr_macd) and is_valid(curr_signal) and is_valid(sma[i]):
            macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            price_above_sma = is_valid(close[i]) and is_valid(sma[i]) and (close[i] > sma[i])

            if macd_cross_up and price_above_sma:
                # Open long position
                order_func.position_open = True
                order_func.entry_index = i
                order_func.highest_since_entry = float(high[i]) if is_valid(high[i]) else float(close[i])
                order_func.entry_price = float(close[i]) if is_valid(close[i]) else np.nan

                # Return order: buy 1 unit (Amount), Long direction
                # size_type=0 (Amount), direction=1 (Long)
                return (1.0, 0, 1)

        # No entry
        return no_order()

    # When in position, update highest_since_entry
    if is_valid(high[i]):
        if high[i] > highest_since_entry:
            order_func.highest_since_entry = float(high[i])
            highest_since_entry = float(high[i])

    # Compute trailing stop if ATR available
    trailing_stop_enabled = is_valid(order_func.highest_since_entry) and is_valid(atr[i])
    trailing_stop = order_func.highest_since_entry - trailing_mult * atr[i] if trailing_stop_enabled else np.inf * -1

    # Exit conditions
    macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal) if is_valid(curr_macd) and is_valid(curr_signal) else False
    price_below_trail = is_valid(close[i]) and trailing_stop_enabled and (close[i] < trailing_stop)

    if macd_cross_down or price_below_trail:
        # Close position
        order_func.position_open = False
        order_func.entry_index = None
        order_func.highest_since_entry = -np.inf
        order_func.entry_price = np.nan

        # Return order: sell 1 unit (Amount), Short direction to close long
        # size_type=0 (Amount), direction=2 (Short)
        return (1.0, 0, 2)

    # Otherwise, do nothing
    return no_order()
