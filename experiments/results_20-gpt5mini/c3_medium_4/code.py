"""
MACD + ATR Trailing Stop Strategy

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- No numba usage
- Uses only pandas and numpy
- Order function returns tuples (size, size_type, direction)
  where size_type and direction are integer codes. We use 0 for Amount and 0 for Both
  (these are the conventional defaults used by the backtest wrapper).

The order_func performs an explicit sequential simulation up to the current index c.i
so decisions are made only with historical data (no lookahead).
"""
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

    Returns a dictionary with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    All returned values are numpy arrays with the same length as the input ohlcv.

    Indicator implementations use only current and past data (no lookahead):
    - MACD: EMA(fast) - EMA(slow) with pandas ewm (adjust=False)
    - Signal: EMA(macd, span=macd_signal)
    - SMA: rolling mean with min_periods=sma_period
    - ATR: Wilder-style EMA of True Range using alpha=1/atr_period
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["close", "high", "low"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    # Work on float copies
    close_s = ohlcv["close"].astype(float).copy()
    high_s = ohlcv["high"].astype(float).copy()
    low_s = ohlcv["low"].astype(float).copy()

    # MACD (EMA fast - EMA slow)
    # Use adjust=False for an online (non-anticipative) EMA
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR
    prev_close = close_s.shift(1)
    tr1 = (high_s - low_s).abs()
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's moving average: alpha = 1/period
    # pandas ewm accepts alpha directly
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # Convert to numpy arrays
    out: Dict[str, np.ndarray] = {
        "close": close_s.values,
        "high": high_s.values,
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
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
    Order function for vectorbt.from_order_func.

    Strategy logic (long-only, single asset):
    - Entry when MACD crosses above Signal AND close > SMA
    - Exit when MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)
    - Trailing stop updates to the highest close seen since entry

    The function performs a sequential simulation from the start up to the current
    index c.i. Decisions at c.i are made using only historical information <= c.i.

    Returns a tuple (size, size_type, direction).
    We use:
      - size: positive to buy, negative to sell
      - size_type: 0 (Amount)
      - direction: 0 (Both)

    When there is no order at the current bar, return (np.nan, 0, 0) so the wrapper
    will translate it to a no-op order.
    """
    # Defensive conversions
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    n = len(close)
    if n == 0:
        return (np.nan, 0, 0)

    # Current bar index
    i = int(getattr(c, "i", 0))
    if i < 0:
        return (np.nan, 0, 0)
    if i >= n:
        # Safety: if called with out-of-bounds index, do nothing
        return (np.nan, 0, 0)

    # Order encoding: (size, size_type, direction)
    SIZE_TYPE_AMOUNT = 0  # Amount
    DIRECTION_BOTH = 0  # Both directions allowed
    ENTRY_SIZE = 1.0
    EXIT_SIZE = -1.0

    # Precompute MACD crosses up/down up to i.
    # We only need values up to i, so index in range(1, i+1).
    def is_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = macd[idx - 1], signal[idx - 1]
        a_curr, b_curr = macd[idx], signal[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def is_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = macd[idx - 1], signal[idx - 1]
        a_curr, b_curr = macd[idx], signal[idx]
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    # Simulate sequentially up to and including index i
    in_position = False
    highest_since_entry = -np.inf

    for t in range(0, i + 1):
        # Entry condition
        if not in_position:
            if is_cross_up(t):
                # Trend filter: price must be above SMA
                if not np.isnan(sma[t]) and not np.isnan(close[t]) and close[t] > sma[t]:
                    # Open long position
                    in_position = True
                    highest_since_entry = close[t] if not np.isnan(close[t]) else -np.inf
                    # If the entry happens at the current bar, return entry order
                    if t == i:
                        return (ENTRY_SIZE, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
                    # else continue simulation to see if exit happens later <= i
            # otherwise no entry
        else:
            # Update highest price since entry using current close
            if not np.isnan(close[t]) and close[t] > highest_since_entry:
                highest_since_entry = close[t]

            # Compute trailing stop threshold for this bar
            threshold = np.nan
            if not np.isnan(highest_since_entry) and not np.isneginf(highest_since_entry) and not np.isnan(atr[t]):
                threshold = highest_since_entry - float(trailing_mult) * atr[t]

            # Exit conditions
            exit_by_macd = is_cross_down(t)
            exit_by_trailing = False
            if not np.isnan(threshold) and not np.isnan(close[t]):
                exit_by_trailing = close[t] < threshold

            if exit_by_macd or exit_by_trailing:
                # Close long position
                in_position = False
                # If the exit happens at the current bar, return exit order
                if t == i:
                    return (EXIT_SIZE, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
                # else continue simulation

    # No order at the current bar
    return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
