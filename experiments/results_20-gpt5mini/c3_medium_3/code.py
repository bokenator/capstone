"""
MACD + ATR Trailing Stop strategy utilities

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- No numba usage.
- Uses vectorbt enums for order returns.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute required indicators for the strategy.

    Returns a dictionary with keys:
    - 'macd': MACD line (fast - slow)
    - 'signal': MACD signal line
    - 'atr': Average True Range
    - 'sma': Simple Moving Average of close
    - 'close': Close prices (numpy array)
    - 'high': High prices (numpy array)

    The function avoids lookahead by only using past and current bars.
    It uses .ewm and .rolling with min_periods=1 to ensure values exist
    early on (so no NaNs after reasonable warmup), while still being causal.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    # Work on copies to avoid modifying user input
    ohlc = ohlcv.copy()

    # Ensure numeric dtype
    close = ohlc["close"].astype(float)
    high = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)

    # MACD: fast and slow EMAs -> MACD line and signal line
    # Use ewm with min_periods=1 to avoid NaNs after warmup
    fast_ema = close.ewm(span=macd_fast, adjust=False, min_periods=1).mean()
    slow_ema = close.ewm(span=macd_slow, adjust=False, min_periods=1).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=macd_signal, adjust=False, min_periods=1).mean()

    # ATR: True range then smoothed with Wilder's method (approximated via ewm)
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=1).mean()

    # SMA trend filter (use min_periods=1 so we have initial values)
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # Convert to numpy arrays for backtester
    return {
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
        "close": close.values,
        "high": high.values,
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
) -> Tuple[float, Any, Any]:
    """
    Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    This implementation is stateless (does not store custom attributes on `c`).
    It simulates the strategy deterministically using only data up to the
    current index (c.i) to decide whether an entry or exit order should be
    emitted at the current bar. This avoids any attribute assignment on the
    provided context object.

    Returns a tuple (size, size_type, direction) or (np.nan, ..., ...) when no order.
    """
    i = int(getattr(c, "i", 0))

    # No-op return (vectorbt wrapper will treat size=np.nan as no order)
    NO_ORDER: Tuple[float, Any, Any] = (float("nan"), SizeType.Amount, Direction.Both)

    # Safety accessor
    def _val(arr: np.ndarray, idx: int) -> float:
        try:
            v = arr[idx]
            # Convert numpy scalar to python float
            if isinstance(v, (np.generic,)):
                v = float(v)
            return v
        except Exception:
            return float("nan")

    # Simulate strategy from t=0..t=i using only past/current data
    in_pos = False
    highest = float("nan")
    entry_at_i = False
    exit_at_i = False

    # Loop over bars up to and including current bar
    for t in range(0, i + 1):
        # Values at t
        close_t = _val(close, t)
        high_t = _val(high, t)
        macd_t = _val(macd, t)
        signal_t = _val(signal, t)
        atr_t = _val(atr, t)
        sma_t = _val(sma, t)

        # Previous values for cross detection
        if t > 0:
            macd_prev = _val(macd, t - 1)
            signal_prev = _val(signal, t - 1)
        else:
            macd_prev = float("nan")
            signal_prev = float("nan")

        cross_up = False
        cross_down = False
        if t > 0 and (not np.isnan(macd_t)) and (not np.isnan(signal_t)) and (not np.isnan(macd_prev)) and (not np.isnan(signal_prev)):
            cross_up = (macd_t > signal_t) and (macd_prev <= signal_prev)
            cross_down = (macd_t < signal_t) and (macd_prev >= signal_prev)

        if not in_pos:
            # Potential entry
            if cross_up and (not np.isnan(sma_t)) and (not np.isnan(close_t)) and (close_t > sma_t):
                in_pos = True
                highest = high_t if not np.isnan(high_t) else close_t
                if t == i:
                    entry_at_i = True
        else:
            # Update highest
            if not np.isnan(high_t):
                if np.isnan(highest):
                    highest = high_t
                else:
                    highest = max(highest, high_t)

            # Compute trailing stop
            trailing_stop = float("nan")
            if (not np.isnan(highest)) and (not np.isnan(atr_t)):
                trailing_stop = highest - float(trailing_mult) * float(atr_t)

            # Check exit conditions
            exited = False
            if cross_down:
                exited = True
            elif (not np.isnan(close_t)) and (not np.isnan(trailing_stop)) and (close_t < trailing_stop):
                exited = True

            if exited:
                in_pos = False
                highest = float("nan")
                if t == i:
                    exit_at_i = True

    # Decide what order to issue at bar i
    # If both entry and exit flagged at same bar, prefer exit (safe choice)
    if exit_at_i:
        # Sell 1 unit to close the long opened earlier (negative size indicates sell when SizeType.Amount is used)
        return (-1.0, SizeType.Amount, Direction.Both)
    if entry_at_i:
        return (1.0, SizeType.Amount, Direction.Both)

    return NO_ORDER
