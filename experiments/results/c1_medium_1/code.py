import numpy as np
import pandas as pd
import vectorbt as vbt
import inspect


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    Order = getattr(vbt.portfolio, 'Order', None)
    info = {}
    info['Order_type'] = str(type(Order))
    try:
        info['Order_dir'] = repr(Order)
    except Exception as e:
        info['Order_dir_err'] = str(e)
    try:
        info['Order_sig'] = str(inspect.signature(Order))
    except Exception as e:
        info['Order_sig_err'] = str(e)
    try:
        info['Order_attrs'] = [a for a in dir(Order) if not a.startswith('_')][:50]
    except Exception as e:
        info['Order_attrs_err'] = str(e)

    raise RuntimeError('DEBUG_ORDER_CONSTRUCTOR:\n' + repr(info))


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    return (np.nan, 0.0, 0, 0)
