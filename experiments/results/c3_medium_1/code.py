import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


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
    if "close" not in ohlcv or "high" not in ohlcv or "low" not in ohlcv:
        raise KeyError("ohlcv DataFrame must contain 'high', 'low' and 'close' columns")
    close_s = ohlcv["close"].astype(float).copy()
    high_s = ohlcv["high"].astype(float).copy()
    low_s = ohlcv["low"].astype(float).copy()
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()
    sma = close_s.rolling(window=sma_period, min_periods=1).mean()
    return {
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr.values,
        "sma": sma.values,
        "close": close_s.values,
        "high": high_s.values,
    }


def order_func(*args: Any, **kwargs: Any) -> Any:
    # DEBUG: raise to inspect the calling convention
    raise RuntimeError(f"DEBUG order_func called with args types: {[type(a) for a in args]}\nargs preview: {args[:10]}\nkwargs keys: {list(kwargs.keys())}")
