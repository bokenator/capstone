import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Args:
        c: vectorbt OrderContext-like object. Expected attributes used:
           - c.i: int, current bar index
           - c.position_now: float, current position size
           - c.cash_now: float, current cash
           - c.pos_record_now: optional position record (numpy.void) containing 'init_i'
        close, high, macd, signal, atr, sma: 1D numpy arrays of prices/indicators
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        Tuple (size, size_type, direction) as per ORDER_CONTEXT_SCHEMA.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Helper: safe access for previous values
    def safe_val(arr: np.ndarray, idx: int) -> float:
        if idx < 0 or idx >= arr.shape[0]:
            return np.nan
        return float(arr[idx]) if not np.isnan(arr[idx]) else np.nan

    # Detect MACD cross up/down with protection against NaNs and i==0
    macd_i = safe_val(macd, i)
    signal_i = safe_val(signal, i)
    macd_prev = safe_val(macd, i - 1)
    signal_prev = safe_val(signal, i - 1)

    cross_up = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_i)
        and not np.isnan(signal_i)
        and macd_prev <= signal_prev
        and macd_i > signal_i
    )

    cross_down = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_i)
        and not np.isnan(signal_i)
        and macd_prev >= signal_prev
        and macd_i < signal_i
    )

    # Trend filter: price above SMA
    sma_i = safe_val(sma, i)
    close_i = safe_val(close, i)
    above_sma = (not np.isnan(sma_i)) and (not np.isnan(close_i)) and (close_i > sma_i)

    # ENTRY: when flat and MACD crosses above Signal AND price > SMA
    if pos == 0.0:
        if cross_up and above_sma:
            # Enter long using 50% of equity (percent size_type=2)
            return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # If we have a position, check exit conditions
    # 1) MACD crosses below Signal
    if pos != 0.0 and cross_down:
        # Close entire long position
        return (-np.inf, 2, 1)

    # 2) Trailing stop based on highest price since entry minus trailing_mult * ATR
    # Determine entry index (init_i) from pos_record_now if available
    highest_since_entry = np.nan
    try:
        pos_record = getattr(c, "pos_record_now", None)
        init_i = None
        if pos_record is not None:
            # Try attribute access then dict-like access
            try:
                init_i = int(getattr(pos_record, "init_i"))
            except Exception:
                try:
                    init_i = int(pos_record["init_i"])  # type: ignore
                except Exception:
                    init_i = None
        if init_i is None:
            # Fallback: assume entry at the beginning (conservative)
            init_i = 0
        init_i = max(0, min(i, int(init_i)))
        # Compute highest high since entry (inclusive)
        if init_i <= i:
            # slice might include NaNs; use nanmax
            highest_since_entry = np.nanmax(high[init_i : i + 1])
    except Exception:
        # Fallback to global max up to current bar
        try:
            highest_since_entry = np.nanmax(high[: i + 1])
        except Exception:
            highest_since_entry = np.nan

    atr_i = safe_val(atr, i)
    stop_price = np.nan
    if not np.isnan(highest_since_entry) and not np.isnan(atr_i):
        stop_price = highest_since_entry - trailing_mult * atr_i

    # If stop_price is valid and current close falls below it, exit
    if not np.isnan(stop_price) and not np.isnan(close_i) and close_i < stop_price:
        return (-np.inf, 2, 1)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute MACD, ATR and SMA indicators required by the strategy using vectorbt.

    Args:
        ohlcv: DataFrame with at least 'high' and 'close' columns. 'low' is optional; if
               missing it will be approximated with 'close' to allow ATR computation.
        macd_fast, macd_slow, macd_signal, sma_period, atr_period: indicator parameters

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'. All values are
        1D numpy arrays of dtype float64 and the same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_series = ohlcv['close'].astype(float)
    high_series = ohlcv['high'].astype(float)

    # Use 'low' if available, otherwise fall back to 'close' (conservative)
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low'].astype(float)
    else:
        low_series = close_series.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values.astype(float)
    signal_arr = macd_ind.signal.values.astype(float)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_arr = atr_ind.atr.values.astype(float)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_arr = sma_ind.ma.values.astype(float)

    # Ensure arrays are 1D and match input length
    n = len(close_series)
    def _ensure_1d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] != n:
            arr = arr.reshape(-1)[:n]
        return arr

    return {
        'close': _ensure_1d(close_series.values),
        'high': _ensure_1d(high_series.values),
        'macd': _ensure_1d(macd_arr),
        'signal': _ensure_1d(signal_arr),
        'atr': _ensure_1d(atr_arr),
        'sma': _ensure_1d(sma_arr),
    }