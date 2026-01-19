import inspect
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple
from collections import namedtuple

# Lightweight Order namedtuple that mimics expected attributes used by vectorbt's order simulation in scalar mode.
# We'll keep a flexible structure initially; later we'll introspect vbt internals to align fields.
Order = namedtuple('Order', ['size', 'price', 'direction', 'size_type'])


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
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    try:
        close_s = ohlcv["close"]
        high_s = ohlcv["high"]
        low_s = ohlcv["low"]
    except Exception as e:
        raise KeyError("ohlcv must contain 'close', 'high', and 'low' columns") from e

    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_arr = atr_ind.atr.values

    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": close_s.values,
        "high": high_s.values,
    }


class _OrderFunc:
    def __init__(self) -> None:
        self.in_position: bool = False
        self.prev_macd: float = np.nan
        self.prev_signal: float = np.nan
        self.highest_since_entry: float = -np.inf
        self._introspected: bool = False

    def __call__(self, *args: Any) -> Any:
        # Introspect vectorbt's replace_inf_price_nb to learn expected order attributes on first call
        if not self._introspected:
            self._introspected = True
            try:
                src = inspect.getsource(vbt.portfolio.nb.replace_inf_price_nb)
            except Exception:
                src = "<could not get source>"
            raise RuntimeError("INTROSPECT_REPLACE_INF_PRICE_NB:\n" + src)

        # The rest of the implementation would go here (omitted during introspection)
        # For safety, return a NoOrder sentinel if called after introspection
        return Order(size=0.0, price=np.inf, direction=0, size_type=0)


_order_func_instance = _OrderFunc()

def order_func(*args: Any) -> Any:
    return _order_func_instance(*args)
