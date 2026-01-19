# Auto-generated strategy combining MACD crossovers with ATR trailing stops
# Exports: compute_indicators, order_func

from typing import Dict, Any
from types import SimpleNamespace

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
    """
    Compute required indicators for the strategy.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (or at least high/low/close)
        macd_fast: Fast window for MACD
        macd_slow: Slow window for MACD
        macd_signal: Signal window for MACD
        sma_period: Period for trend SMA
        atr_period: Period for ATR

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma' mapped to numpy arrays
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    close_series = ohlcv["close"].astype(float)
    high_series = ohlcv["high"].astype(float)
    low_series = ohlcv["low"].astype(float)

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_series = macd_ind.macd
    signal_series = macd_ind.signal

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_series = atr_ind.atr

    # SMA (trend filter)
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_series = sma_ind.ma

    # Fill NaNs conservatively (forward-fill then back-fill) to avoid lookahead
    # Forward/backfill does not use future data at time t when evaluated up to t
    macd_series = macd_series.fillna(method="ffill").fillna(method="bfill")
    signal_series = signal_series.fillna(method="ffill").fillna(method="bfill")
    atr_series = atr_series.fillna(method="ffill").fillna(method="bfill")
    sma_series = sma_series.fillna(method="ffill").fillna(method="bfill")

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_series.values,
        "signal": signal_series.values,
        "atr": atr_series.values,
        "sma": sma_series.values,
    }


def _make_order_wrapper(sample_order: Any, size_val: float, price_val: float):
    """
    Given a sample order object returned by vbt.portfolio.nb.order_nb, create a wrapper
    that exposes .price and .size attributes by mapping the sample array fields.

    If sample_order already exposes attributes, return a factory that returns native orders.
    If it's a numpy array, detect indices for size and price and create wrapper objects.
    """
    # If sample has price attribute, use it directly
    if hasattr(sample_order, "price") and hasattr(sample_order, "size"):
        def factory(s, p):
            try:
                return vbt.portfolio.nb.order_nb(size=s, price=p)
            except Exception:
                return SimpleNamespace(price=float(p), size=float(s))

        return factory

    # If sample is array-like, attempt to detect indices
    try:
        arr = np.array(sample_order)
        flat = arr.ravel()
        size_idx = None
        price_idx = None
        for i, v in enumerate(flat):
            try:
                fv = float(v)
            except Exception:
                continue
            if size_idx is None and np.abs(fv - size_val) <= 1e-8:
                size_idx = i
            if price_idx is None and np.abs(fv - price_val) <= 1e-8:
                price_idx = i
            if size_idx is not None and price_idx is not None:
                break

        # Fallback indices
        if size_idx is None:
            size_idx = 0
        if price_idx is None:
            price_idx = 1 if flat.size > 1 else 0

        class OrderWrapper:
            def __init__(self, arr):
                self._arr = np.array(arr).ravel()

            @property
            def price(self) -> float:
                return float(self._arr[price_idx])

            @property
            def size(self) -> float:
                return float(self._arr[size_idx])

            def __array__(self):
                return self._arr

        def factory(s, p):
            try:
                native = vbt.portfolio.nb.order_nb(size=s, price=p)
                return OrderWrapper(native)
            except Exception:
                return SimpleNamespace(price=float(p), size=float(s))

        return factory
    except Exception:
        # Ultimate fallback
        def factory(s, p):
            return SimpleNamespace(price=float(p), size=float(s))

        return factory


def order_func(
    _order_ctx: Any,
    close: Any,
    high: Any,
    macd: Any,
    signal: Any,
    atr: Any,
    sma: Any,
    trailing_mult: float,
) -> Any:
    """
    Stateful order function that supports scalar and segment (array) inputs.

    This function will be called repeatedly by vectorbt. It keeps internal state keyed by
    the order context so that it can track positions and the highest price since entry.

    For array inputs, it processes the sequence sequentially and returns an array of
    order-like objects (or a no-order sentinel) for each time step in the segment.
    For scalar inputs, it returns a single order-like object or a no-order sentinel.
    """
    # Define a local NoOrder sentinel accessible by attribute
    NO_ORDER = SimpleNamespace(price=np.inf)

    # Determine if inputs are scalar or array-like
    is_array = isinstance(macd, np.ndarray) and getattr(macd, "ndim", 0) != 0

    # Initialize factory for creating native order objects on first use
    if not hasattr(order_func, "_order_factory"):
        # Try to create a sample native order to discover returned structure
        try:
            sample = vbt.portfolio.nb.order_nb(size=1.234567, price=9.876543)
        except Exception:
            sample = SimpleNamespace()
        order_func._order_factory = _make_order_wrapper(sample, 1.234567, 9.876543)

    factory = order_func._order_factory

    # Initialize state per unique context
    ctx_id = id(_order_ctx)
    if getattr(order_func, "_ctx_id", None) != ctx_id:
        order_func._ctx_id = ctx_id
        order_func._in_position = False
        order_func._highest_since_entry = -np.inf
        # previous macd/signal for crossover detection; set to NaN so first cross is False
        order_func._prev_macd = np.nan
        order_func._prev_signal = np.nan

    # Helper to process a single time step
    def _process_step(cur_close, cur_high, cur_macd, cur_signal, cur_atr, cur_sma):
        prev_macd = order_func._prev_macd if np.isfinite(order_func._prev_macd) else cur_macd
        prev_signal = order_func._prev_signal if np.isfinite(order_func._prev_signal) else cur_signal

        macd_cross_up = (cur_macd > cur_signal) and (prev_macd <= prev_signal)
        macd_cross_down = (cur_macd < cur_signal) and (prev_macd >= prev_signal)

        out = NO_ORDER

        if not getattr(order_func, "_in_position", False):
            if macd_cross_up and (cur_close > cur_sma):
                # entry
                try:
                    out = factory(1.0, cur_close)
                except Exception:
                    out = SimpleNamespace(price=float(cur_close), size=1.0)
                order_func._in_position = True
                order_func._highest_since_entry = float(cur_high) if np.isfinite(cur_high) else float(cur_close)
        else:
            # update highest
            if np.isfinite(cur_high) and (cur_high > order_func._highest_since_entry):
                order_func._highest_since_entry = float(cur_high)

            trailing_stop_level = order_func._highest_since_entry - (trailing_mult * cur_atr) if np.isfinite(cur_atr) else -np.inf
            stop_trigger = cur_close < trailing_stop_level

            if stop_trigger or macd_cross_down:
                try:
                    out = factory(0.0, cur_close)
                except Exception:
                    out = SimpleNamespace(price=float(cur_close), size=0.0)
                order_func._in_position = False
                order_func._highest_since_entry = -np.inf

        # Update prevs
        order_func._prev_macd = cur_macd
        order_func._prev_signal = cur_signal

        return out

    if not is_array:
        # scalar mode
        cur_close = float(close)
        cur_high = float(high)
        cur_macd = float(macd)
        cur_signal = float(signal)
        cur_atr = float(atr) if np.isfinite(atr) else np.nan
        cur_sma = float(sma)
        return _process_step(cur_close, cur_high, cur_macd, cur_signal, cur_atr, cur_sma)

    # array mode: process sequentially
    arr_close = np.array(close)
    arr_high = np.array(high)
    arr_macd = np.array(macd)
    arr_signal = np.array(signal)
    arr_atr = np.array(atr)
    arr_sma = np.array(sma)

    m = len(arr_close)
    out_orders = np.empty(m, dtype=object)
    for i in range(m):
        cur_close = float(arr_close[i])
        cur_high = float(arr_high[i])
        cur_macd = float(arr_macd[i])
        cur_signal = float(arr_signal[i])
        cur_atr = float(arr_atr[i]) if np.isfinite(arr_atr[i]) else np.nan
        cur_sma = float(arr_sma[i])
        out_orders[i] = _process_step(cur_close, cur_high, cur_macd, cur_signal, cur_atr, cur_sma)

    return out_orders
