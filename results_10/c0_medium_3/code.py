"""
MACD + ATR Trailing Stop Strategy implementation for vectorbt backtester.

Exports:
- compute_indicators
- order_func

Notes:
- Do NOT use numba.
- Returns simple Python tuples from order_func: (size, size_type, direction)

Strategy:
- MACD (fast=12, slow=26, signal=9)
- Trend filter: 50-period SMA
- ATR period: 14
- Trailing stop = highest_since_entry - trailing_mult * ATR (trailing_mult passed to order_func)
- Entry: MACD cross up and price > SMA
- Exit: MACD cross down OR price < trailing_stop

This file is intended to be passed to the provided run_backtest() harness.
"""

from typing import Any, Dict, Tuple

import math

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
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] (at least high/low/close).
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for the trend SMA filter.
        atr_period: Period for ATR.

    Returns:
        Dict with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    # Ensure numeric dtype and copy series
    close_s = ohlcv["close"].astype(float).copy()
    high_s = ohlcv["high"].astype(float).copy()
    low_s = ohlcv["low"].astype(float).copy()

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd)
    # Use pandas' ewm (adjust=False) for common MACD implementation.
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter) - require full window to be considered valid (min_periods = sma_period)
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR: True Range then Wilder's smoothing (EMA with alpha=1/period)
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # Convert to numpy arrays of float (ensures np.nan where invalid)
    return {
        "close": close_s.values.astype(float),
        "high": high_s.values.astype(float),
        "macd": macd_line.values.astype(float),
        "signal": signal_line.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
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
    Order function used by vectorbt.Portfolio.from_order_func (use_numba=False).

    Returns a tuple (size, size_type, direction). To indicate no action, return (np.nan, 0, 0).

    Implementation notes:
    - Long-only strategy. We buy 1 unit on entry and sell -1 unit on exit (size_type=0 -> Amount, direction=0 -> Both).
    - Maintains simple Python-side state on the function object to track whether we're in a position
      and the highest high since entry for the ATR trailing stop.

    Args:
        c: callback context provided by vectorbt (expects c.i to index current bar).
        close, high, macd, signal, atr, sma: numpy arrays produced by compute_indicators.
        trailing_mult: Multiplier for ATR used in trailing stop (e.g., 2.0).

    Returns:
        Tuple[size (float), size_type (int), direction (int)].
    """
    # Get current index
    i = int(getattr(c, "i", 0))

    # Reset state at the beginning of the run
    if i == 0 or not hasattr(order_func, "_state_initialized"):
        order_func._in_position = False
        order_func._highest = -np.inf
        order_func._entry_index = None
        order_func._state_initialized = True

    # Helper to safely get a float value (handles numpy types and NaNs)
    def _safe_float(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[int(idx)])
        except Exception:
            return float("nan")

    # Current and previous values (use safe float conversions)
    cur_macd = _safe_float(macd, i)
    cur_signal = _safe_float(signal, i)
    cur_close = _safe_float(close, i)
    cur_high = _safe_float(high, i)
    cur_atr = _safe_float(atr, i)
    cur_sma = _safe_float(sma, i)

    prev_macd = _safe_float(macd, i - 1) if i - 1 >= 0 else float("nan")
    prev_signal = _safe_float(signal, i - 1) if i - 1 >= 0 else float("nan")

    # Detect MACD cross up / down safely (require non-nan values)
    cross_up = False
    cross_down = False
    if not math.isnan(prev_macd) and not math.isnan(prev_signal) and not math.isnan(cur_macd) and not math.isnan(cur_signal):
        cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
        cross_down = (prev_macd >= prev_signal) and (cur_macd < cur_signal)

    # If not in position, check entry conditions
    if not getattr(order_func, "_in_position", False):
        # Entry: MACD cross up AND price above SMA
        if cross_up and (not math.isnan(cur_sma)) and (not math.isnan(cur_close)) and (cur_close > cur_sma):
            # Enter long: buy 1 unit (size_type=0: Amount, direction=0: Both)
            order_func._in_position = True
            order_func._entry_index = i
            # Initialize highest with current high
            order_func._highest = cur_high if not math.isnan(cur_high) else -np.inf
            return (1.0, 0, 0)

        # No action
        return (np.nan, 0, 0)

    # If in a position, update highest high since entry
    if not math.isnan(cur_high):
        if math.isnan(order_func._highest) or order_func._highest == -np.inf:
            order_func._highest = cur_high
        else:
            order_func._highest = max(order_func._highest, cur_high)

    # Compute trailing stop level if ATR and highest are available
    trailing_level = float("nan")
    if (order_func._highest is not None) and (order_func._highest != -np.inf) and (not math.isnan(cur_atr)):
        trailing_level = order_func._highest - float(trailing_mult) * cur_atr

    # Exit conditions: MACD cross down OR price falls below trailing stop
    if cross_down:
        # Exit full position: sell 1 unit
        order_func._in_position = False
        order_func._highest = -np.inf
        order_func._entry_index = None
        return (-1.0, 0, 0)

    if not math.isnan(trailing_level) and (not math.isnan(cur_close)) and (cur_close < trailing_level):
        # Exit full position: sell 1 unit
        order_func._in_position = False
        order_func._highest = -np.inf
        order_func._entry_index = None
        return (-1.0, 0, 0)

    # Otherwise, do nothing
    return (np.nan, 0, 0)
