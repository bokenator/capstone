import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict
from collections import namedtuple

# Basic fallback order namedtuple used if vectorbt's Order type isn't found
_FallbackOrder = namedtuple("FallbackOrder", ["size", "price", "size_type", "direction"])


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Any:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array (use close[c.i] for current price)
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        An order object compatible with vectorbt's simulation. The factory to build the
        appropriate Order type is resolved at runtime and cached on the function.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent state on first call or when a new run starts (i == 0)
    if (not hasattr(order_func, "_in_pos")) or i == 0:
        order_func._in_pos = False
        order_func._entry_idx = -1
        order_func._highest = np.nan

    # On first invocation, try to resolve vectorbt's Order type and build a factory
    if not hasattr(order_func, "_make_order"):
        OrderType = None
        # Try common locations where Order namedtuple/type may be defined
        try:
            OrderType = getattr(vbt.portfolio.nb, "Order")
        except Exception:
            OrderType = None
        if OrderType is None:
            try:
                OrderType = getattr(vbt.portfolio.enums, "Order")
            except Exception:
                OrderType = None
        if OrderType is None:
            try:
                OrderType = getattr(vbt.portfolio.base, "Order")
            except Exception:
                OrderType = None

        # If OrderType is a namedtuple-like with _fields, prepare a factory that fills required fields
        if getattr(OrderType, "_fields", None) is not None:
            fields = list(OrderType._fields)

            def _make_order(size: float, size_type: int, direction: int):
                kwargs: Dict[str, Any] = {}
                for f in fields:
                    if f == "size":
                        kwargs[f] = float(size) if not np.isnan(size) else np.nan
                    elif f == "price":
                        kwargs[f] = np.inf
                    elif f == "size_type":
                        kwargs[f] = int(size_type)
                    elif f == "direction":
                        kwargs[f] = int(direction)
                    elif f == "fees":
                        kwargs[f] = 0.0
                    elif f == "slippage":
                        kwargs[f] = 0.0
                    elif f == "max_size":
                        # Ensure a positive max_size so execution logic works
                        if not np.isnan(size) and not np.isinf(size):
                            try:
                                kwargs[f] = float(abs(size))
                            except Exception:
                                kwargs[f] = 1e12
                        else:
                            kwargs[f] = 1e12
                    elif f == "size_granularity":
                        kwargs[f] = 1.0
                    elif f == "min_size":
                        kwargs[f] = 0.0
                    else:
                        # Generic default values: prefer numeric types
                        kwargs[f] = 0.0
                try:
                    return OrderType(**kwargs)
                except Exception:
                    # Fallback to simple namedtuple if instantiation fails
                    return _FallbackOrder(size=float(size) if not np.isnan(size) else np.nan,
                                           price=np.inf,
                                           size_type=int(size_type),
                                           direction=int(direction))

        else:
            # OrderType not found - fallback to simple namedtuple
            def _make_order(size: float, size_type: int, direction: int):
                return _FallbackOrder(size=float(size) if not np.isnan(size) else np.nan,
                                       price=np.inf,
                                       size_type=int(size_type),
                                       direction=int(direction))

        order_func._make_order = _make_order

    make_order = order_func._make_order

    # Helper functions
    def _prev(arr: np.ndarray, idx: int):
        if idx <= 0:
            return np.nan
        return arr[idx - 1]

    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        prev_macd = _prev(macd, idx)
        prev_sig = _prev(signal, idx)
        cur_macd = macd[idx]
        cur_sig = signal[idx]
        if np.isnan(prev_macd) or np.isnan(prev_sig) or np.isnan(cur_macd) or np.isnan(cur_sig):
            return False
        return (prev_macd <= prev_sig) and (cur_macd > cur_sig)

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        prev_macd = _prev(macd, idx)
        prev_sig = _prev(signal, idx)
        cur_macd = macd[idx]
        cur_sig = signal[idx]
        if np.isnan(prev_macd) or np.isnan(prev_sig) or np.isnan(cur_macd) or np.isnan(cur_sig):
            return False
        return (prev_macd >= prev_sig) and (cur_macd < cur_sig)

    # Current bar values
    cur_close = float(close[i]) if not np.isnan(close[i]) else np.nan
    cur_high = float(high[i]) if not np.isnan(high[i]) else np.nan
    cur_sma = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    cur_atr = float(atr[i]) if not np.isnan(atr[i]) else np.nan

    # Sync state if portfolio indicates a position but our state does not
    if pos != 0 and not getattr(order_func, "_in_pos", False):
        order_func._in_pos = True
        order_func._entry_idx = i
        order_func._highest = cur_high if not np.isnan(cur_high) else cur_close

    # ENTRY
    if pos == 0 or pos == 0.0:
        if macd_cross_up(i) and (not np.isnan(cur_sma)) and (not np.isnan(cur_close)) and (cur_close > cur_sma):
            # Compute number of shares to buy using ~99% of available cash
            available_cash = float(c.cash_now) if hasattr(c, "cash_now") else np.nan
            if not np.isnan(available_cash) and available_cash > 0 and not np.isnan(cur_close) and cur_close > 0:
                size_shares = float(np.floor((available_cash * 0.99) / cur_close))
                if size_shares <= 0:
                    # Nothing to buy
                    return make_order(np.nan, 0, 0)
                order_func._in_pos = True
                order_func._entry_idx = i
                order_func._highest = cur_high if not np.isnan(cur_high) else cur_close
                return make_order(size_shares, 0, 1)
            else:
                return make_order(np.nan, 0, 0)

    else:
        # Update highest
        if not np.isnan(cur_high):
            if np.isnan(order_func._highest) or (cur_high > order_func._highest):
                order_func._highest = cur_high

        # MACD bearish cross exit
        if macd_cross_down(i):
            # Close entire long position by specifying negative of current position
            size_to_close = -float(pos)
            order_func._in_pos = False
            order_func._entry_idx = -1
            order_func._highest = np.nan
            return make_order(size_to_close, 0, 1)

        # Trailing stop exit
        if not np.isnan(order_func._highest) and not np.isnan(cur_atr):
            trailing_level = order_func._highest - (trailing_mult * cur_atr)
            if (not np.isnan(cur_close)) and (cur_close < trailing_level):
                size_to_close = -float(pos)
                order_func._in_pos = False
                order_func._entry_idx = -1
                order_func._highest = np.nan
                return make_order(size_to_close, 0, 1)

    # No action
    return make_order(np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators. Use vectorbt indicator classes.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)

    # If low is missing, fall back to close for ATR calculation
    if "low" in ohlcv.columns:
        low = ohlcv["low"].astype(float)
    else:
        low = close

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values.astype(float)
    signal_arr = macd_ind.signal.values.astype(float)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    atr_arr = atr_ind.atr.values.astype(float)

    # Compute SMA (simple moving average) as trend filter
    sma_ind = vbt.MA.run(close, window=sma_period)
    sma_arr = sma_ind.ma.values.astype(float)

    return {
        "close": close.values.astype(float),
        "high": high.values.astype(float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }
