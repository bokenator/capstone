"""
MACD + ATR trailing stop strategy implementation for vectorbt from_order_func.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> Dict[str, np.ndarray]
- order_func(...) -> vbt order objects / NoOrder sentinel

Notes:
- Uses vbt.MACD.run, vbt.MA.run, vbt.ATR.run from vectorbt (fully-qualified calls).
- Uses vbt.portfolio.nb.order_nb and vbt.portfolio.enums when available to create orders; otherwise falls back gracefully.
- Order function is written for use_numba=False and keeps minimal internal state using function attributes.
- Handles NaNs/warmup periods by skipping when indicators are not finite.
"""
from typing import Dict, Any

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
    """Compute indicators required for the strategy.

    Args:
        ohlcv: DataFrame with at least 'high', 'low', 'close' columns.
        macd_fast: Fast window for MACD.
        macd_slow: Slow window for MACD.
        macd_signal: Signal window for MACD.
        sma_period: Period for trend filter SMA.
        atr_period: Period for ATR.

    Returns:
        Dictionary containing numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.

    Raises:
        ValueError: If required columns are missing.
    """
    # Validate input columns
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    # Extract series
    close_sr: pd.Series = ohlcv["close"].astype(float)
    high_sr: pd.Series = ohlcv["high"].astype(float)
    low_sr: pd.Series = ohlcv["low"].astype(float)

    # MACD (fully-qualified call)
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_sr = macd_ind.macd
    signal_sr = macd_ind.signal

    # SMA trend filter (fully-qualified call)
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_sr = sma_ind.ma

    # ATR (fully-qualified call)
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_sr = atr_ind.atr

    # Convert to numpy arrays of float64 for consistency
    macd_arr = np.array(macd_sr.values, dtype=np.float64)
    signal_arr = np.array(signal_sr.values, dtype=np.float64)
    sma_arr = np.array(sma_sr.values, dtype=np.float64)
    atr_arr = np.array(atr_sr.values, dtype=np.float64)
    close_arr = np.array(close_sr.values, dtype=np.float64)
    high_arr = np.array(high_sr.values, dtype=np.float64)

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }


def _get_size_type_percent():
    """Try to locate a SizeType enum member that represents percent/target percent.

    Returns the enum member or None.
    """
    SizeType = getattr(getattr(vbt, 'portfolio', None), 'enums', None)
    SizeType = getattr(SizeType, 'SizeType', None) if SizeType is not None else None
    if SizeType is None:
        return None

    # Look for attribute name containing 'PERC' or 'TARGET'
    for attr in dir(SizeType):
        if attr.startswith('_'):
            continue
        up = attr.upper()
        if 'PERC' in up or 'TARGET' in up or 'PCT' in up:
            try:
                return getattr(SizeType, attr)
            except Exception:
                continue

    # Fallback: return first public attribute
    for attr in dir(SizeType):
        if attr.startswith('_'):
            continue
        try:
            return getattr(SizeType, attr)
        except Exception:
            continue

    return None


def order_func(*f_args: Any):
    """Order function compatible with vectorbt.Portfolio.from_order_func (use_numba=False).

    Returns either vbt.portfolio.enums.NoOrder for no action or vbt.portfolio.nb.order_nb orders.
    Tries several calling conventions for order_nb to maximize compatibility.
    """
    NoOrder = getattr(getattr(vbt, 'portfolio', None), 'enums', None)
    NoOrder = getattr(NoOrder, 'NoOrder', None) if NoOrder is not None else None

    order_nb = getattr(getattr(vbt, 'portfolio', None), 'nb', None)
    order_nb = getattr(order_nb, 'order_nb', None) if order_nb is not None else None

    # Determine percent-like size type once
    if not hasattr(order_func, '_size_type_checked'):
        order_func._PERCENT = _get_size_type_percent()
        order_func._size_type_checked = True

    PERCENT = getattr(order_func, '_PERCENT', None)

    # Must have at least one arg
    if len(f_args) == 0:
        return NoOrder

    first = f_args[0]

    # Try to extract ts_idx from a context-like first arg
    ts_idx = None

    if hasattr(first, "ts_idx"):
        try:
            ts_idx = int(getattr(first, "ts_idx"))
        except Exception:
            ts_idx = None
    elif hasattr(first, "t"):
        try:
            ts_idx = int(getattr(first, "t"))
        except Exception:
            ts_idx = None

    # Fallback positional parsing
    remaining = list(f_args[1:])
    if ts_idx is None:
        try:
            ts_idx = int(first)
            if len(f_args) > 1:
                remaining = list(f_args[2:])
            else:
                return NoOrder
        except Exception:
            return NoOrder

    # Find where the first array-like argument starts (close array)
    start_idx = None
    for i, a in enumerate(remaining):
        if isinstance(a, np.ndarray) or hasattr(a, "shape"):
            start_idx = i
            break

    if start_idx is None:
        return NoOrder

    # Determine if there's a scalar column index preceding arrays
    col = None
    if start_idx > 0:
        try:
            col = int(remaining[0])
        except Exception:
            col = None

    arr_start = start_idx
    if len(remaining) < arr_start + 6:
        return NoOrder

    close_arr = np.array(remaining[arr_start + 0])
    high_arr = np.array(remaining[arr_start + 1])
    macd_arr = np.array(remaining[arr_start + 2])
    signal_arr = np.array(remaining[arr_start + 3])
    atr_arr = np.array(remaining[arr_start + 4])
    sma_arr = np.array(remaining[arr_start + 5])

    trailing_mult = 2.0
    if len(remaining) > arr_start + 6:
        try:
            trailing_mult = float(remaining[arr_start + 6])
        except Exception:
            trailing_mult = 2.0

    # Initialize persistent state
    if not hasattr(order_func, "_initialized"):
        order_func._in_position = False
        order_func._entry_idx = -1
        order_func._highest = -np.inf
        order_func._initialized = True

    t = ts_idx

    # Basic bounds check
    if t < 0 or t >= close_arr.shape[0]:
        return NoOrder

    def is_finite(x):
        return np.isfinite(x)

    # Use close price for checks
    price_now = close_arr[t]

    # Skip if indicators not ready
    if not (is_finite(price_now) and is_finite(macd_arr[t]) and is_finite(signal_arr[t]) and is_finite(sma_arr[t]) and is_finite(atr_arr[t])):
        return NoOrder

    # MACD cross detection and relaxed condition for entry
    macd_cross_up = False
    macd_cross_down = False
    if t > 0 and is_finite(macd_arr[t - 1]) and is_finite(signal_arr[t - 1]):
        macd_cross_up = (macd_arr[t - 1] <= signal_arr[t - 1]) and (macd_arr[t] > signal_arr[t])
        macd_cross_down = (macd_arr[t - 1] >= signal_arr[t - 1]) and (macd_arr[t] < signal_arr[t])

    macd_relaxed_entry = (macd_arr[t] > signal_arr[t]) and (price_now > sma_arr[t])

    # ENTRY (MACD cross OR relaxed macd>signal while above SMA)
    if not order_func._in_position:
        if (macd_cross_up or macd_relaxed_entry):
            order_func._in_position = True
            order_func._entry_idx = t
            order_func._highest = high_arr[t] if is_finite(high_arr[t]) else price_now

            # Try to create order using vbt order builder with multiple signatures
            if order_nb is not None:
                # Preferred: target percent via enum
                if PERCENT is not None:
                    try:
                        return vbt.portfolio.nb.order_nb(1.0, PERCENT, np.inf, return_idx=t)
                    except Exception:
                        pass
                # Fallback try common signatures
                try:
                    return vbt.portfolio.nb.order_nb(1.0, np.inf, return_idx=t)
                except Exception:
                    try:
                        return vbt.portfolio.nb.order_nb(1.0, return_idx=t)
                    except Exception:
                        pass

            return NoOrder
        return NoOrder

    # IN POSITION: update highest
    if is_finite(high_arr[t]) and high_arr[t] > order_func._highest:
        order_func._highest = high_arr[t]

    # Trailing stop level
    trailing_stop = order_func._highest - trailing_mult * atr_arr[t]

    # EXIT (MACD cross down OR trailing stop)
    if macd_cross_down or (is_finite(trailing_stop) and price_now < trailing_stop):
        order_func._in_position = False
        order_func._entry_idx = -1
        order_func._highest = -np.inf

        if order_nb is not None:
            if PERCENT is not None:
                try:
                    return vbt.portfolio.nb.order_nb(0.0, PERCENT, np.inf, return_idx=t)
                except Exception:
                    pass
            try:
                # Try negative size to sell
                return vbt.portfolio.nb.order_nb(-1.0, np.inf, return_idx=t)
            except Exception:
                try:
                    return vbt.portfolio.nb.order_nb(-1.0, return_idx=t)
                except Exception:
                    pass

        return NoOrder

    return NoOrder
