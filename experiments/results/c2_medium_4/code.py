# Temporary debug version: will raise an exception listing enum attributes to help mapping
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
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    close_sr = ohlcv["close"].copy()
    high_sr = ohlcv["high"].copy()
    low_sr = ohlcv["low"].copy()
    macd_res = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_sr = macd_res.macd
    signal_sr = macd_res.signal
    sma_sr = vbt.MA.run(close_sr, window=sma_period).ma
    atr_sr = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period).atr
    def clean_series(s):
        return s.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
    return {
        "close": clean_series(close_sr).values,
        "high": clean_series(high_sr).values,
        "macd": clean_series(macd_sr).values,
        "signal": clean_series(signal_sr).values,
        "atr": clean_series(atr_sr).values,
        "sma": clean_series(sma_sr).values,
    }


def order_func(order_ctx: Any, close: np.ndarray, high: np.ndarray, macd: np.ndarray, signal: np.ndarray, atr: np.ndarray, sma: np.ndarray, trailing_mult: float) -> Tuple[Any, ...]:
    from vectorbt.portfolio.enums import Direction, SizeType, NoOrder
    # Debug: list attributes
    dir_attrs = [a for a in dir(Direction) if not a.startswith('_')]
    size_attrs = [a for a in dir(SizeType) if not a.startswith('_')]
    # Raise exception to surface attributes
    raise Exception(f"Direction attributes: {dir_attrs}; SizeType attributes: {size_attrs}")
    return NoOrder
