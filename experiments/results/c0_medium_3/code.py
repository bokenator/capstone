# Test returning a simple object with .price attribute instead of tuple/namedtuple

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class SimpleOrder:
    def __init__(self, size: float, price: float = np.inf):
        self.size = float(size)
        self.price = float(price)
        # Fill other attributes used by vectorbt if accessed
        self.size_type = 0
        self.direction = 2
        self.fees = 0.0
        self.fixed_fees = 0.0
        self.slippage = 0.0
        self.min_size = 0.0
        self.max_size = np.inf
        self.size_granularity = np.nan
        self.reject_prob = 0.0
        self.lock_cash = False
        self.allow_partial = True
        self.raise_reject = False
        self.log = False

    def __repr__(self):
        return f"SimpleOrder(size={self.size}, price={self.price})"


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    sma = close.rolling(window=sma_period, min_periods=1).mean()

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    return {"macd": macd.values, "signal": signal.values, "atr": atr.values, "sma": sma.values, "close": close.values, "high": high.values}


def order_func(
    ctx: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
    *args,
) -> Tuple[Any, ...]:
    try:
        i = int(getattr(ctx, "i"))
    except Exception:
        i = int(getattr(ctx, "index", 0))

    price = float(close[i])
    # For test, always place a single buy at first bar
    if i == 1:
        return SimpleOrder(size=1.0, price=price)
    return ()
