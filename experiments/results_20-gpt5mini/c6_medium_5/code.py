# Strategy: MACD crossover entries with ATR-based trailing stops
# Exports:
# - compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> dict[str, np.ndarray]
# - order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple(size, size_type, direction)

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    All values are 1D numpy arrays of the same length as the input ohlcv.

    Notes:
    - Uses vectorbt.MACD.run and vectorbt.ATR.run to compute MACD and ATR.
    - SMA is computed via pandas rolling mean.
    - Does not fill NaNs with future data (no lookahead). Some NaNs may
      remain in the initial warmup period.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close_s = ohlcv["close"]
    high_s = ohlcv["high"]
    low_s = ohlcv["low"]

    # MACD: vectorbt.MACD.run
    macd_ind = vbt.MACD.run(
        close_s,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR: vectorbt.ATR.run
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # SMA: pandas rolling mean (no forward-filling)
    sma_s = close_s.rolling(window=sma_period).mean()

    # Extract numpy arrays (1D)
    macd_arr = np.asarray(macd_ind.macd).ravel()
    signal_arr = np.asarray(macd_ind.signal).ravel()
    atr_arr = np.asarray(atr_ind.atr).ravel()
    sma_arr = np.asarray(sma_s).ravel()
    close_arr = np.asarray(close_s).ravel()
    high_arr = np.asarray(high_s).ravel()

    # Ensure all arrays have same length
    n = len(close_arr)
    for arr in (high_arr, macd_arr, signal_arr, atr_arr, sma_arr):
        if len(arr) != n:
            raise ValueError("Indicator arrays must match input length")

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }


def _find_entry_index_backwards(i: int, macd: np.ndarray, signal: np.ndarray, close: np.ndarray, sma: np.ndarray) -> int:
    """Fallback: find the most recent index <= i where an entry MACD crossover + SMA condition happened.

    Only looks at data up to index i (no lookahead).
    Returns 0 if nothing found.
    """
    for j in range(i, 0, -1):
        # require previous value exists
        if j - 1 < 0:
            continue
        m_prev = macd[j - 1]
        s_prev = signal[j - 1]
        m_curr = macd[j]
        s_curr = signal[j]

        # skip invalid/missing values
        if np.isnan(m_prev) or np.isnan(s_prev) or np.isnan(m_curr) or np.isnan(s_curr):
            continue

        if (m_prev <= s_prev) and (m_curr > s_curr) and (not np.isnan(close[j])) and (not np.isnan(sma[j])) and (close[j] > sma[j]):
            return j
    return 0


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """Order function for vbt.Portfolio.from_order_func (Python, non-numba).

    Returns a tuple: (size, size_type, direction)
      - size: float (np.nan for no action)
      - size_type: int (we use TargetPercent = 5)
      - direction: int (LongOnly = 0)

    Logic:
      - Entry: MACD crosses above signal (t-1 <= t) AND close > SMA AND not in position
      - Exit: MACD crosses below OR close < (highest_since_entry - trailing_mult * ATR)

    Important:
      - Uses c.i for current index and c.position_now for current position size
      - Tries to read entry index from c.pos_record_now.init_i (if available)
        otherwise falls back to scanning backwards to find the entry bar.
    """
    i = int(c.i)
    # Default: no action
    NO_ACTION = (float("nan"), 0, 0)

    # Safety checks
    if i <= 0:
        return NO_ACTION

    # Current values
    try:
        macd_prev = float(macd[i - 1])
        signal_prev = float(signal[i - 1])
        macd_curr = float(macd[i])
        signal_curr = float(signal[i])
    except Exception:
        return NO_ACTION

    price = float(close[i])
    sma_val = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # Position status
    try:
        position_now = float(c.position_now)
    except Exception:
        # If unavailable, assume not in position
        position_now = 0.0

    # SizeType.TargetPercent = 5, Direction.LongOnly = 0
    TARGET_PERCENT = 5
    LONG_ONLY = 0

    # ENTRY: MACD bullish cross AND price above SMA
    if position_now <= 0:
        # require MACD cross up
        if (
            (not np.isnan(macd_prev))
            and (not np.isnan(signal_prev))
            and (not np.isnan(macd_curr))
            and (not np.isnan(signal_curr))
            and (macd_prev <= signal_prev)
            and (macd_curr > signal_curr)
            and (not np.isnan(price))
            and (not np.isnan(sma_val))
            and (price > sma_val)
        ):
            # Enter full allocation (100% of portfolio)
            return (1.0, TARGET_PERCENT, LONG_ONLY)
        return NO_ACTION

    # If here, we are in a long position: check exit conditions
    # 1) MACD bearish cross
    if (
        (not np.isnan(macd_prev))
        and (not np.isnan(signal_prev))
        and (not np.isnan(macd_curr))
        and (not np.isnan(signal_curr))
        and (macd_prev >= signal_prev)
        and (macd_curr < signal_curr)
    ):
        return (0.0, TARGET_PERCENT, LONG_ONLY)

    # 2) Trailing stop based on highest price since entry and ATR
    init_i = None
    try:
        # Try to read entry index from pos_record_now (numpy.void-like)
        pos_rec = c.pos_record_now
        # Try attribute access first
        init_i_attr = getattr(pos_rec, "init_i", None)
        if init_i_attr is not None:
            init_i = int(init_i_attr)
        else:
            # Try dict-like access
            try:
                init_i = int(pos_rec["init_i"])
            except Exception:
                init_i = None
    except Exception:
        init_i = None

    # Fallback: scan backwards to find the most recent entry index
    if init_i is None or init_i < 0 or init_i > i:
        try:
            init_i = _find_entry_index_backwards(i, macd, signal, close, sma)
        except Exception:
            init_i = 0

    # Compute highest high since entry (inclusive)
    try:
        if init_i < 0:
            init_i = 0
        high_since = np.nanmax(high[init_i : i + 1])
    except Exception:
        high_since = np.nanmax(high[: i + 1])

    atr_val = float(atr[i]) if not np.isnan(atr[i]) else np.nan

    if (not np.isnan(high_since)) and (not np.isnan(atr_val)):
        trail_price = float(high_since - trailing_mult * atr_val)
        if price < trail_price:
            return (0.0, TARGET_PERCENT, LONG_ONLY)

    # No action
    return NO_ACTION
