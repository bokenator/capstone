# Complete implementation for MACD + ATR trailing stop strategy using vectorbt

from typing import Dict, Any

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
    """
    Compute indicators required by the strategy.

    Returns a dictionary with numpy arrays for keys:
    - 'macd': MACD line
    - 'signal': MACD signal line
    - 'atr': Average True Range
    - 'sma': Simple moving average
    - 'close': close prices (np.ndarray)
    - 'high': high prices (np.ndarray)

    Parameters are typed and default to the strategy specification.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame with columns: 'open','high','low','close',...")
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise KeyError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_series = macd_ind.macd.fillna(method="bfill").fillna(0.0)
    signal_series = macd_ind.signal.fillna(method="bfill").fillna(0.0)

    # ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    atr_series = atr_ind.atr.fillna(method="bfill").fillna(0.0)

    # SMA trend filter
    sma_ind = vbt.MA.run(close, window=sma_period)
    sma_series = sma_ind.ma.fillna(method="bfill").fillna(0.0)

    return {
        "macd": macd_series.values,
        "signal": signal_series.values,
        "atr": atr_series.values,
        "sma": sma_series.values,
        "close": close.values,
        "high": high.values,
    }


# Helpers to pick enum members with fallbacks

def _get_enum_member(enum_cls: Any, candidates: list, default: Any) -> Any:
    for name in candidates:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return default


def _find_enum_member_by_keywords(enum_cls: Any, keywords: list, default: Any) -> Any:
    for name in dir(enum_cls):
        if name.startswith("_"):
            continue
        up = name.upper()
        if any(kw in up for kw in keywords):
            return getattr(enum_cls, name)
    return default


def order_func(*args: Any) -> Any:
    """
    Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    This implementation expects the last 7 arguments in *args to be:
      (close, high, macd, signal, atr, sma, trailing_mult)

    The function maintains minimal internal state (call index, position flag, highest price since entry)
    using function attributes. It issues a buy of 1 unit on entry and a sell of 1 unit on exit.

    Returns an order object (created by vbt.portfolio.nb.order_nb) or vbt.portfolio.enums.NoOrder for no order.
    """
    # We expect at least the order args to be present at the tail of args
    ORDER_ARGS_COUNT = 7
    if len(args) < ORDER_ARGS_COUNT:
        raise ValueError(
            f"order_func requires at least {ORDER_ARGS_COUNT} trailing arguments: (close, high, macd, signal, atr, sma, trailing_mult)"
        )

    # Extract the trailing order arguments (these are the ones passed to from_order_func)
    close_arr = np.array(args[-ORDER_ARGS_COUNT + 0])
    high_arr = np.array(args[-ORDER_ARGS_COUNT + 1])
    macd_arr = np.array(args[-ORDER_ARGS_COUNT + 2])
    signal_arr = np.array(args[-ORDER_ARGS_COUNT + 3])
    atr_arr = np.array(args[-ORDER_ARGS_COUNT + 4])
    sma_arr = np.array(args[-ORDER_ARGS_COUNT + 5])
    trailing_mult = float(args[-ORDER_ARGS_COUNT + 6])

    # Initialize persistent state on first invocation
    if not hasattr(order_func, "_initialized"):
        order_func._call_idx = 0
        order_func._in_position = False
        order_func._highest = np.nan
        order_func._entry_index = -1
        order_func._initialized = True

    i = int(order_func._call_idx)

    # If index exceeds array length, return NoOrder
    if i >= len(close_arr):
        return vbt.portfolio.enums.NoOrder

    # Current bar values
    close_i = float(close_arr[i])
    high_i = float(high_arr[i])
    macd_i = float(macd_arr[i])
    signal_i = float(signal_arr[i])
    atr_i = float(atr_arr[i])
    sma_i = float(sma_arr[i])

    # Previous bar values (safely handle i == 0)
    if i > 0:
        macd_prev = float(macd_arr[i - 1])
        signal_prev = float(signal_arr[i - 1])
    else:
        macd_prev = macd_i
        signal_prev = signal_i

    macd_cross_up = (macd_prev <= signal_prev) and (macd_i > signal_i)
    macd_cross_down = (macd_prev >= signal_prev) and (macd_i < signal_i)

    # Pick enum members (fallback to ints if members not present)
    size_type_amount = _get_enum_member(vbt.portfolio.enums.SizeType, ["Amount", "Size", "Fixed"], 0)
    dir_long = _find_enum_member_by_keywords(vbt.portfolio.enums.Direction, ["LONG", "BUY", "B"], 1)
    dir_short = _find_enum_member_by_keywords(vbt.portfolio.enums.Direction, ["SHORT", "SELL", "S"], -1)
    dir_both = _find_enum_member_by_keywords(vbt.portfolio.enums.Direction, ["BOTH"], None)

    dir_entry = dir_long
    dir_exit = dir_both if dir_both is not None else dir_long

    # Entry: not currently in position, MACD cross up and price above SMA
    if not order_func._in_position:
        if macd_cross_up and np.isfinite(close_i) and np.isfinite(sma_i) and (close_i > sma_i):
            order_func._in_position = True
            order_func._entry_index = i
            order_func._highest = high_i if np.isfinite(high_i) else close_i
            order_func._call_idx += 1
            # Create a market buy order for 1 unit (use current close as price)
            try:
                return vbt.portfolio.nb.order_nb(size=1.0, size_type=size_type_amount, direction=dir_entry, price=close_i)
            except TypeError:
                # Fallback: try different ordering of args (size, price)
                try:
                    return vbt.portfolio.nb.order_nb(1.0, close_i)
                except Exception:
                    # As a last resort, return NoOrder to avoid crashing
                    return vbt.portfolio.enums.NoOrder
        else:
            order_func._call_idx += 1
            return vbt.portfolio.enums.NoOrder

    # If in position, update highest and evaluate exits
    if order_func._in_position:
        if np.isfinite(high_i):
            if not np.isfinite(order_func._highest):
                order_func._highest = high_i
            else:
                order_func._highest = max(order_func._highest, high_i)

        trailing_stop = np.nan
        if np.isfinite(order_func._highest) and np.isfinite(atr_i):
            trailing_stop = order_func._highest - trailing_mult * atr_i

        exit_by_trail = False
        if np.isfinite(trailing_stop):
            exit_by_trail = close_i < trailing_stop

        exit_by_macd = macd_cross_down

        if exit_by_macd or exit_by_trail:
            # Exit position: sell 1 unit
            order_func._in_position = False
            order_func._highest = np.nan
            order_func._entry_index = -1
            order_func._call_idx += 1
            try:
                return vbt.portfolio.nb.order_nb(size=1.0, size_type=size_type_amount, direction=dir_exit, price=close_i)
            except TypeError:
                try:
                    return vbt.portfolio.nb.order_nb(-1.0, close_i)
                except Exception:
                    return vbt.portfolio.enums.NoOrder

    # No order
    order_func._call_idx += 1
    return vbt.portfolio.enums.NoOrder
