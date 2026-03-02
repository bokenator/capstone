# Complete strategy implementation combining MACD crossover entries with ATR-based trailing stops.
# Exports: compute_indicators, order_func

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
) -> pd.DataFrame:
    """
    Compute indicators required by the strategy.

    Returns a DataFrame with columns: close, high, macd, signal, atr, sma.

    Notes:
    - MACD is computed as EMA(fast) - EMA(slow) with ewm(adjust=False), signal is EMA of MACD.
    - SMA uses a standard rolling mean with min_periods=sma_period to produce NaNs before warmup.
    - ATR uses True Range and Wilder-style EMA (ewm with alpha=1/period, adjust=False).

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least 'high','low','close').
        macd_fast, macd_slow, macd_signal, sma_period, atr_period: indicator parameters.

    Returns:
        DataFrame indexed same as ohlcv with required columns.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns or 'low' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high', 'low' and 'close' columns")

    close = ohlcv['close'].astype(float).copy()
    high = ohlcv['high'].astype(float).copy()
    low = ohlcv['low'].astype(float).copy()

    # MACD
    # Use ewm with adjust=False for recursive EMA
    fast_ema = close.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close.ewm(span=macd_slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing approximation using ewm with alpha=1/period
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # Build DataFrame
    indicators = pd.DataFrame(
        {
            'close': close,
            'high': high,
            'macd': macd,
            'signal': signal,
            'atr': atr,
            'sma': sma,
        },
        index=ohlcv.index,
    )

    return indicators


def order_func(
    c: Any,
    close: Any,
    high: Any,
    macd: Any,
    signal: Any,
    atr: Any,
    sma: Any,
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """
    Order function to be used with vectorbt.Portfolio.from_order_func (use_numba=False).

    Strategy logic (long-only):
    - Entry when MACD crosses above Signal and price > SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction).
    - size: target size (use target sizing so 1.0 means fully long, 0.0 means flat)
    - size_type: integer enum for size_type. We use 2 as a reasonable mapping for 'Target' sizing.
    - direction: integer enum for direction. We use 1 for 'Long'. These integers are compatible
      with typical vectorbt enums (Direction.Long == 1).

    Notes:
    - The function only uses data up to c.i (no lookahead).
    - When no action is required, return (np.nan, 0, 0) so the wrapper treats it as no order.
    """
    # Convert inputs to numpy arrays for fast indexing; they may be pandas Series
    close_arr = np.asarray(close)
    high_arr = np.asarray(high)
    macd_arr = np.asarray(macd)
    signal_arr = np.asarray(signal)
    atr_arr = np.asarray(atr)
    sma_arr = np.asarray(sma)

    i = int(getattr(c, 'i', 0))  # current integer index

    # Constants for returned enums (integers). We avoid importing vectorbt enums per instructions.
    SIZE_TYPE_TARGET = 2  # assumed mapping for SizeType.Target
    DIRECTION_LONG = 1    # assumed mapping for Direction.Long

    # Safety: if index is out of bounds or indicators have NaN at current index, do nothing
    n = len(close_arr)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Avoid making decisions on the very first bar
    if i == 0:
        return (np.nan, 0, 0)

    # Helper to safely get previous values (handles NaNs)
    def _safe_get(arr: np.ndarray, idx: int, default: float = np.nan) -> float:
        try:
            v = arr[idx]
            if np.isnan(v):
                return default
            return float(v)
        except Exception:
            return default

    # Evaluate MACD crossover at current index (uses t-1 and t values)
    macd_prev = _safe_get(macd_arr, i - 1)
    signal_prev = _safe_get(signal_arr, i - 1)
    macd_curr = _safe_get(macd_arr, i)
    signal_curr = _safe_get(signal_arr, i)

    # Price and SMA at current index
    price_curr = _safe_get(close_arr, i)
    sma_curr = _safe_get(sma_arr, i)

    # Entry condition: MACD crosses above Signal AND price > SMA
    macd_cross_up = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_curr)
        and not np.isnan(signal_curr)
        and (macd_prev <= signal_prev)
        and (macd_curr > signal_curr)
    )
    entry_condition = macd_cross_up and (not np.isnan(sma_curr)) and (price_curr > sma_curr)

    # Determine whether we are currently in a position using the context
    in_position = False
    entry_idx = None
    try:
        # c.position is available in vectorbt order funcs; it represents the current position
        pos = getattr(c, 'position', None)
        if pos is not None:
            # Several attributes may exist; check for is_open or opened
            in_position = bool(getattr(pos, 'is_open', False) or getattr(pos, 'opened', False))

            # Try to get entry index
            # Different vectorbt versions use different attribute names; try common ones
            if in_position:
                if hasattr(pos, 'entry_idx'):
                    entry_idx = int(pos.entry_idx)
                elif hasattr(pos, 'entry_index'):
                    entry_idx = int(pos.entry_index)
                elif hasattr(pos, 'entries'):
                    # entries may be an array; take the last
                    try:
                        ent = pos.entries
                        if len(ent) > 0:
                            entry_idx = int(ent[-1])
                    except Exception:
                        entry_idx = None
    except Exception:
        # If anything fails, fallback to not in position and compute entry index by scanning
        in_position = False
        entry_idx = None

    # If not in position but entry condition holds -> place entry order (target=1.0 long)
    if (not in_position) and entry_condition:
        return (1.0, SIZE_TYPE_TARGET, DIRECTION_LONG)

    # If in position, check exit conditions
    if in_position:
        # If we couldn't get entry_idx from context, try to infer from past indicators deterministically
        if entry_idx is None:
            # Find all past indices where entry condition would have been True (based only on past data)
            # We evaluate cross using safe arrays up to current index inclusive
            if i >= 1:
                prev_macd = macd_arr[: i + 1].copy()
                prev_signal = signal_arr[: i + 1].copy()
                prev_close = close_arr[: i + 1].copy()
                prev_sma = sma_arr[: i + 1].copy()

                # Compute crosses for indices 1..i
                crosses = np.full(i + 1, False, dtype=bool)
                for j in range(1, i + 1):
                    if (
                        not np.isnan(prev_macd[j - 1])
                        and not np.isnan(prev_signal[j - 1])
                        and not np.isnan(prev_macd[j])
                        and not np.isnan(prev_signal[j])
                        and (prev_macd[j - 1] <= prev_signal[j - 1])
                        and (prev_macd[j] > prev_signal[j])
                    ):
                        if not np.isnan(prev_sma[j]) and (prev_close[j] > prev_sma[j]):
                            crosses[j] = True
                idxs = np.where(crosses)[0]
                if idxs.size > 0:
                    entry_idx = int(idxs[-1])
                else:
                    # If no prior detected entry, assume earliest bar
                    entry_idx = 0
            else:
                entry_idx = 0

        # Ensure entry_idx is within bounds
        entry_idx = int(max(0, min(entry_idx, i)))

        # Compute highest price since entry (use high prices when available)
        try:
            highest_since_entry = float(np.nanmax(high_arr[entry_idx : i + 1]))
        except Exception:
            highest_since_entry = float(np.nanmax(close_arr[entry_idx : i + 1]))

        atr_curr = _safe_get(atr_arr, i)
        # If ATR is NaN, skip trailing stop check
        trailing_threshold = np.nan
        if not np.isnan(atr_curr):
            trailing_threshold = highest_since_entry - (trailing_mult * atr_curr)

        # MACD cross down condition
        macd_cross_down = (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and not np.isnan(macd_curr)
            and not np.isnan(signal_curr)
            and (macd_prev >= signal_prev)
            and (macd_curr < signal_curr)
        )

        # Exit if MACD bearish cross or price falls below trailing stop
        trailing_hit = (not np.isnan(trailing_threshold)) and (price_curr < trailing_threshold)

        if macd_cross_down or trailing_hit:
            # Set target to 0.0 to exit
            return (0.0, SIZE_TYPE_TARGET, DIRECTION_LONG)

    # Default: no order
    return (np.nan, 0, 0)
