# Update: use TargetPercent size type to set target exposure (1.0 => 100%, 0.0 => 0%) and Direction.LongOnly

from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import linregress


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    for col in ("high", "low", "close"):
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    close_s = ohlcv["close"]
    high_s = ohlcv["high"]
    low_s = ohlcv["low"]

    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd
    signal_line = macd_ind.signal

    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_series = atr_ind.atr

    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_series = sma_ind.ma

    return {
        "close": close_s.values.astype(np.float64),
        "high": high_s.values.astype(np.float64),
        "macd": macd_line.values.astype(np.float64),
        "signal": signal_line.values.astype(np.float64),
        "atr": atr_series.values.astype(np.float64),
        "sma": sma_series.values.astype(np.float64),
    }


def _as_1d_array(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return np.array([float(obj)])
        if obj.ndim == 1:
            return obj.astype(np.float64)
        try:
            return obj.ravel().astype(np.float64)
        except Exception:
            return None
    if isinstance(obj, pd.Series):
        return obj.values.astype(np.float64)
    try:
        arr = np.array(obj, dtype=np.float64)
        if arr.ndim == 1:
            return arr
    except Exception:
        pass
    return None


def order_func(*args: Any):
    # Default no-order sentinel
    NO_ORDER = vbt.portfolio.enums.NoOrder

    arrays = []
    scalar_after_last_array = None
    last_array_idx = -1
    for idx, a in enumerate(args):
        arr = _as_1d_array(a)
        if arr is not None and arr.size > 1:
            arrays.append(arr)
            last_array_idx = idx

    if last_array_idx >= 0 and last_array_idx < len(args) - 1:
        for a in args[last_array_idx + 1 :]:
            if isinstance(a, (float, int)):
                scalar_after_last_array = float(a)
                break

    if len(arrays) < 6:
        return NO_ORDER

    close_arr, high_arr, macd_arr, signal_arr, atr_arr, sma_arr = arrays[-6:]
    trailing_mult = float(scalar_after_last_array) if scalar_after_last_array is not None else 2.0

    n = len(close_arr)

    if (not hasattr(order_func, "_initialized")) or (getattr(order_func, "_close_len", None) != n):
        order_func._initialized = True
        order_func._call_idx = 0
        order_func._close_len = n
        order_func.position = 0
        order_func.entry_index = -1
        order_func.highest_since_entry = -np.inf

    t = int(order_func._call_idx)

    if t >= n:
        return NO_ORDER

    price = float(close_arr[t]) if np.isfinite(close_arr[t]) else np.nan

    def get_val(arr: np.ndarray, idx: int) -> float:
        if idx < 0 or idx >= len(arr):
            return float("nan")
        return float(arr[idx])

    macd_i = get_val(macd_arr, t)
    signal_i = get_val(signal_arr, t)
    macd_prev = get_val(macd_arr, t - 1)
    signal_prev = get_val(signal_arr, t - 1)
    sma_i = get_val(sma_arr, t)
    atr_i = get_val(atr_arr, t)

    if order_func.position == 1 and np.isfinite(price):
        order_func.highest_since_entry = max(order_func.highest_since_entry, price)

    bull_cross = (
        np.isfinite(macd_prev)
        and np.isfinite(signal_prev)
        and macd_prev <= signal_prev
        and np.isfinite(macd_i)
        and np.isfinite(signal_i)
        and macd_i > signal_i
    )

    bear_cross = (
        np.isfinite(macd_prev)
        and np.isfinite(signal_prev)
        and macd_prev >= signal_prev
        and np.isfinite(macd_i)
        and np.isfinite(signal_i)
        and macd_i < signal_i
    )

    # SizeType.TargetPercent == 5 according to introspection
    SIZE_TARGET_PERCENT = vbt.portfolio.enums.SizeType.TargetPercent
    DIR_LONG_ONLY = vbt.portfolio.enums.Direction.LongOnly

    # Ensure price valid before creating an order
    if not np.isfinite(price) or price <= 0:
        order_func._call_idx = t + 1
        return NO_ORDER

    if order_func.position == 0:
        if bull_cross and np.isfinite(sma_i) and price > sma_i:
            order_func.position = 1
            order_func.entry_index = t
            order_func.highest_since_entry = price
            order_func._call_idx = t + 1
            # Target 100% exposure
            return vbt.portfolio.nb.order_nb(1.0, price, SIZE_TARGET_PERCENT, DIR_LONG_ONLY)
        else:
            order_func._call_idx = t + 1
            return NO_ORDER

    if order_func.position == 1:
        trailing_stop_triggered = False
        if np.isfinite(order_func.highest_since_entry) and np.isfinite(atr_i) and np.isfinite(price):
            stop_price = order_func.highest_since_entry - float(trailing_mult) * float(atr_i)
            trailing_stop_triggered = price < stop_price

        macd_exit = bear_cross

        if trailing_stop_triggered or macd_exit:
            order_func.position = 0
            order_func.entry_index = -1
            order_func.highest_since_entry = -np.inf
            order_func._call_idx = t + 1
            # Target 0% exposure to close position
            return vbt.portfolio.nb.order_nb(0.0, price, SIZE_TARGET_PERCENT, DIR_LONG_ONLY)

    order_func._call_idx = t + 1
    return NO_ORDER
