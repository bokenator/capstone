# Generated strategy combining MACD crossover entries with ATR-based trailing stops
# Exports:
# - compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
#     -> returns a pandas.DataFrame with columns: close, high, macd, signal, atr, sma
# - order_func(c, close, high, macd, signal, atr, sma, trailing_mult)
#     -> order function usable by vectorbt.Portfolio.from_order_func

from __future__ import annotations

from typing import Union, Tuple

import numpy as np
import pandas as pd

# Global state store for order contexts to avoid assigning arbitrary attributes to the context
_ORDER_STATE: dict[int, dict] = {}


def compute_indicators(
    ohlcv: Union[pd.DataFrame, pd.Series, np.ndarray],
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute indicators required by the strategy.

    Args:
        ohlcv: Data input. Can be a DataFrame with columns ['open','high','low','close','volume'],
               a Series/1D array interpreted as close prices, or a DataFrame with at least
               'close' column. If high/low are missing, they are copied from close.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal EMA period for MACD.
        sma_period: Period for trend filter SMA.
        atr_period: Period for ATR.

    Returns:
        pd.DataFrame with columns: close, high, macd, signal, atr, sma.
        Length and index match the input.

    Notes:
        - All moving averages are computed in a causal way (no lookahead) using
          pandas ewm/rolling with adjust=False where applicable.
        - Uses min_periods=1 for SMA/ATR so indicators are available from the start.
    """

    # Normalize input to pandas Series/DataFrame
    if isinstance(ohlcv, pd.DataFrame):
        df_in = ohlcv.copy()
        # Try to find close/high/low
        if 'close' in df_in.columns:
            close = df_in['close'].astype(float)
        elif 'Close' in df_in.columns:
            close = df_in['Close'].astype(float)
        else:
            # If DataFrame but no close column, attempt to use first column
            close = df_in.iloc[:, 0].astype(float)

        if 'high' in df_in.columns:
            high = df_in['high'].astype(float)
        elif 'High' in df_in.columns:
            high = df_in['High'].astype(float)
        else:
            high = close.copy()

        if 'low' in df_in.columns:
            low = df_in['low'].astype(float)
        elif 'Low' in df_in.columns:
            low = df_in['Low'].astype(float)
        else:
            low = close.copy()

        idx = df_in.index
    else:
        # ohlcv is Series or array-like: treat as close
        close = pd.Series(ohlcv).astype(float)
        high = close.copy()
        low = close.copy()
        idx = close.index if hasattr(close, 'index') else pd.RangeIndex(start=0, stop=len(close))

    # Ensure pandas Series with proper index
    close = pd.Series(close.values, index=idx, dtype='float64')
    high = pd.Series(high.values, index=idx, dtype='float64')
    low = pd.Series(low.values, index=idx, dtype='float64')

    # MACD: EMA fast - EMA slow, signal is EMA of MACD. Use adjust=False to avoid lookahead.
    # Use span parameter for ewm which is common for MACD.
    fast_ema = close.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close.ewm(span=macd_slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR: True Range then rolling mean (min_periods=1 so ATR available from the start)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # Assemble DataFrame
    indicators = pd.DataFrame(
        {
            'close': close,
            'high': high,
            'macd': macd,
            'signal': signal,
            'atr': atr,
            'sma': sma,
        },
        index=idx,
    )

    return indicators


def order_func(
    c,
    close: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    macd: Union[np.ndarray, pd.Series],
    signal: Union[np.ndarray, pd.Series],
    atr: Union[np.ndarray, pd.Series],
    sma: Union[np.ndarray, pd.Series],
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """
    Order function implementing:
      - Entry: MACD crosses above signal AND price > 50-period SMA
      - Exit: MACD crosses below signal OR price drops below (highest_since_entry - trailing_mult * ATR)

    This is a long-only strategy.

    The function stores per-context state in a global dict keyed by id(c) to avoid setting
    arbitrary attributes on the provided context object.

    Returns tuple (size, size_type, direction) where size may be np.nan to indicate no order.
    """

    # Constants for SizeType and Direction (integers used by vectorbt order_nb)
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_BOTH = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    # Convert inputs to numpy arrays for fast indexing
    close_a = np.asarray(close)
    high_a = np.asarray(high)
    macd_a = np.asarray(macd)
    signal_a = np.asarray(signal)
    atr_a = np.asarray(atr)
    sma_a = np.asarray(sma)

    i = int(c.i) if hasattr(c, 'i') else 0

    # Use a global state dict keyed by id(c)
    ctx_id = id(c)
    if ctx_id not in _ORDER_STATE:
        _ORDER_STATE[ctx_id] = {
            'in_pos': False,
            'highest': float('-inf'),
            'entry_i': -1,
            'trailing_stop': float('inf'),
        }
    st = _ORDER_STATE[ctx_id]

    # Default: no order
    no_order = (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Safety: ensure we have enough data
    n = len(close_a)
    if i >= n:
        return no_order

    # Current values
    close_i = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    macd_i = float(macd_a[i]) if not np.isnan(macd_a[i]) else np.nan
    signal_i = float(signal_a[i]) if not np.isnan(signal_a[i]) else np.nan
    atr_i = float(atr_a[i]) if not np.isnan(atr_a[i]) else np.nan
    sma_i = float(sma_a[i]) if not np.isnan(sma_a[i]) else np.nan

    # Previous values for crossover detection
    if i == 0:
        macd_prev = np.nan
        signal_prev = np.nan
    else:
        macd_prev = float(macd_a[i - 1]) if not np.isnan(macd_a[i - 1]) else np.nan
        signal_prev = float(signal_a[i - 1]) if not np.isnan(signal_a[i - 1]) else np.nan

    # ENTRY: MACD bullish cross AND price above SMA
    bullish_cross = False
    try:
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            bullish_cross = (macd_prev <= signal_prev) and (macd_i > signal_i)
    except Exception:
        bullish_cross = False

    if (not st['in_pos']) and bullish_cross and (not np.isnan(sma_i)) and (not np.isnan(close_i)) and (close_i > sma_i):
        # Enter long: buy 1 unit (amount)
        st['in_pos'] = True
        st['entry_i'] = i
        st['highest'] = close_i
        # Compute trailing stop based on current ATR if available
        if not np.isnan(atr_i):
            st['trailing_stop'] = st['highest'] - trailing_mult * atr_i
        else:
            st['trailing_stop'] = float('-inf')

        return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # If in position, update highest and check exit conditions
    if st['in_pos']:
        # Update highest price since entry
        if not np.isnan(close_i) and close_i > st['highest']:
            st['highest'] = close_i
            # Update trailing stop when new high achieved and ATR available
            if not np.isnan(atr_i):
                st['trailing_stop'] = st['highest'] - trailing_mult * atr_i

        # Check MACD bearish cross (exit)
        bearish_cross = False
        try:
            if not np.isnan(macd_prev) and not np.isnan(signal_prev):
                bearish_cross = (macd_prev >= signal_prev) and (macd_i < signal_i)
        except Exception:
            bearish_cross = False

        if bearish_cross:
            # Exit by selling 1 unit (negative amount)
            st['in_pos'] = False
            st['highest'] = float('-inf')
            st['trailing_stop'] = float('inf')
            return (-1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

        # Check trailing stop breach
        if (st.get('trailing_stop', None) is not None) and (st['trailing_stop'] not in [None, float('inf')]):
            # Use current ATR to compute threshold if ATR available, else use stored trailing stop
            threshold = st['highest'] - trailing_mult * atr_i if (not np.isnan(atr_i)) else st['trailing_stop']
            # If price falls strictly below threshold -> exit
            if (not np.isnan(close_i)) and (not np.isnan(threshold)) and (close_i < threshold):
                st['in_pos'] = False
                st['highest'] = float('-inf')
                st['trailing_stop'] = float('inf')
                return (-1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # No action
    return no_order
