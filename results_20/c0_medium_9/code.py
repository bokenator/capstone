"""
MACD + ATR Trailing Stop Strategy

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- No numba usage.
- order_func uses a precomputed target-percent array (cached) and emits discrete amount orders:
  - Buy 1.0 unit on entry (SizeType.Amount)
  - Sell 1.0 unit on exit (negative amount with Direction.Both)

Assumptions about enum integers (avoids importing vbt.enums):
- SIZE_TYPE_AMOUNT = 0
- SIZE_TYPE_TARGET_PERCENT = 2 (kept for reference if needed)
- Direction mapping (observed in many vectorbt versions):
    DIRECTION_LONG = 0
    DIRECTION_SHORT = 1
    DIRECTION_BOTH = 2

These are educated guesses aligned with vectorbt examples used elsewhere.
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

    Returns a dictionary with numpy arrays for keys:
    - close, high, macd, signal, atr, sma

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (case-insensitive assumed)
        macd_fast, macd_slow, macd_signal: MACD ema periods
        sma_period: period for trend filter SMA
        atr_period: period for ATR

    Returns:
        dict[str, np.ndarray]
    """
    # Validate input contains required columns
    required_cols = {"open", "high", "low", "close"}
    cols_lower = {c.lower() for c in ohlcv.columns}
    if not required_cols.issubset(cols_lower):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    # Normalize column names to lower-case access
    df = ohlcv.copy()

    def _get(col: str) -> pd.Series:
        # find the actual column name ignoring case
        for c in df.columns:
            if c.lower() == col:
                return pd.to_numeric(df[c], errors="coerce")
        raise KeyError(col)

    close_s = _get("close").astype(float)
    high_s = _get("high").astype(float)
    low_s = _get("low").astype(float)

    # MACD: fast EMA - slow EMA; signal is EMA of MACD
    # Use adjust=False for typical EMA implementation
    ema_fast = close_s.ewm(span=macd_fast, adjust=False, min_periods=1).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False, min_periods=1).mean()
    macd_series = ema_fast - ema_slow
    signal_series = macd_series.ewm(span=macd_signal, adjust=False, min_periods=1).mean()

    # SMA trend filter
    sma_series = close_s.rolling(window=sma_period, min_periods=1).mean()

    # ATR calculation (classic): rolling mean of True Range
    prev_close = close_s.shift(1)
    tr1 = (high_s - low_s).abs()
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=atr_period, min_periods=1).mean()

    # Return numpy arrays
    return {
        "close": close_s.to_numpy(dtype=float),
        "high": high_s.to_numpy(dtype=float),
        "macd": macd_series.to_numpy(dtype=float),
        "signal": signal_series.to_numpy(dtype=float),
        "atr": atr_series.to_numpy(dtype=float),
        "sma": sma_series.to_numpy(dtype=float),
    }


# Constants for sizing / direction (avoid importing vbt.enums)
SIZE_TYPE_AMOUNT = 0
SIZE_TYPE_TARGET_PERCENT = 2
# Direction mapping guessed to match vectorbt's Direction enum
DIRECTION_LONG = 0
DIRECTION_SHORT = 1
DIRECTION_BOTH = 2


def _compute_target_percent(
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> np.ndarray:
    """
    Compute target percent (1.0 for long, 0.0 for flat) for each bar based on
    MACD crossover entries and ATR-based trailing stop.

    This is done by simulating the strategy deterministically over the series.
    """
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    n = len(close)
    if not (len(high) == n == len(macd) == len(signal) == len(atr) == len(sma)):
        raise ValueError("Indicator arrays must have the same length")

    # Prepare previous values for cross detection
    macd_prev = np.concatenate(([np.nan], macd[:-1]))
    signal_prev = np.concatenate(([np.nan], signal[:-1]))

    valid_prev = (~np.isnan(macd_prev)) & (~np.isnan(signal_prev)) & (~np.isnan(macd)) & (~np.isnan(signal))
    cross_up = (macd_prev <= signal_prev) & (macd > signal) & valid_prev
    cross_down = (macd_prev >= signal_prev) & (macd < signal) & valid_prev

    target = np.zeros(n, dtype=float)

    in_pos = False
    current_high = -np.inf

    for i in range(n):
        # Evaluate entry
        if not in_pos:
            can_enter = False
            if cross_up[i]:
                # require price above sma for trend filter
                if (not np.isnan(close[i])) and (not np.isnan(sma[i])):
                    if close[i] > sma[i]:
                        can_enter = True
            if can_enter:
                in_pos = True
                current_high = high[i] if not np.isnan(high[i]) else -np.inf
                target[i] = 1.0
            else:
                target[i] = 0.0
        else:
            # update highest since entry
            if not np.isnan(high[i]) and high[i] > current_high:
                current_high = high[i]

            # compute stop price if ATR available
            stop_price = np.nan
            if not np.isnan(atr[i]) and not np.isinf(current_high):
                stop_price = current_high - float(trailing_mult) * float(atr[i])

            exit_by_stop = False
            if not np.isnan(stop_price) and not np.isnan(close[i]):
                exit_by_stop = close[i] < stop_price

            if cross_down[i] or exit_by_stop:
                in_pos = False
                current_high = -np.inf
                target[i] = 0.0
            else:
                target[i] = 1.0

    return target


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
    Order function for vectorbt.from_order_func (python mode, no numba).

    This implementation emits discrete amount orders (1 unit) on entry and -1 unit on exit.
    It caches the computed target-percent array on the function object to avoid recomputing it
    on every call.
    """
    # Ensure numpy arrays
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    # Compute and cache target percent array on the function to avoid recomputation
    cache_attr = "_target_percent_cache"
    cache_len_attr = "_target_percent_cache_len"
    if not hasattr(order_func, cache_attr) or getattr(order_func, cache_len_attr, None) != len(close):
        target = _compute_target_percent(close, high, macd, signal, atr, sma, trailing_mult)
        setattr(order_func, cache_attr, target)
        setattr(order_func, cache_len_attr, len(close))
    else:
        target = getattr(order_func, cache_attr)

    # Current index from context
    idx = getattr(c, "i", None)
    if idx is None:
        # No context index -> no action
        return (np.nan, 0, 0)

    # Bound index
    if idx < 0 or idx >= len(target):
        return (np.nan, 0, 0)

    curr = float(target[int(idx)])
    prev = float(target[int(idx - 1)]) if int(idx) > 0 else 0.0

    # Entry: issue a buy order for 1 unit
    if curr > prev:
        return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # Exit: issue a sell order for 1 unit (negative amount). Use Direction.Both for closing
    if curr < prev:
        # Use negative amount to indicate selling 1 unit
        return (-1.0, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # No action
    return (np.nan, 0, 0)
