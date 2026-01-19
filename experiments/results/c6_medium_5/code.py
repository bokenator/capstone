import inspect
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Optional

# Import order helpers from vectorbt (allowed by VAS)
from vectorbt.portfolio.nb import order_nb
from vectorbt.portfolio.enums import NoOrder, Direction, SizeType

# Debug: inspect order_nb signature and enums
raise RuntimeError(f"order_nb sig={inspect.signature(order_nb)}, Direction members={dir(Direction)}, SizeType members={dir(SizeType)}, NoOrder={NoOrder}")


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    pass


def order_func(*args: Any) -> Any:
    return NoOrder
