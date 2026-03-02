"""
RSI Mean Reversion Signal Generator

Generates long-only position targets based on RSI mean reversion:
- RSI period = 14 (default, can be overridden via params)
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70

Exports:
- generate_signals(data, params) -> dict[str, pd.Series]

The function expects `data` to be a dict with key 'ohlcv' mapping to a
pandas.DataFrame that contains a 'close' column.

Uses vectorbt's RSI indicator for calculation.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv'
            mapping to a pandas DataFrame with a 'close' column.
        params: Strategy parameters. Supported keys:
            - rsi_window (int): RSI lookback period. Default 14.
            - rsi_oversold (float): Oversold threshold. Default 30.0.
            - rsi_overbought (float): Overbought threshold. Default 70.0.
            - rsi_ewm (bool): Whether to use EWM-based RSI. Default False.

    Returns:
        A dictionary with key 'ohlcv' mapping to a pandas Series of position targets
        (1 = long, 0 = flat). The Series is aligned with the input close prices' index.

    Raises:
        ValueError: If input data is malformed.
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame of OHLCV data")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"]

    # Parameters with defaults
    rsi_window = int(params.get("rsi_window", 14))
    rsi_oversold = float(params.get("rsi_oversold", 30.0))
    rsi_overbought = float(params.get("rsi_overbought", 70.0))
    rsi_ewm = bool(params.get("rsi_ewm", False))

    # Compute RSI using vectorbt
    # vbt.RSI.run returns an object with attribute `.rsi` (Series)
    rsi_res = vbt.RSI.run(close, window=rsi_window, ewm=rsi_ewm)
    rsi = rsi_res.rsi

    # Align RSI with close index (safety) and ensure float dtype
    rsi = rsi.reindex(close.index).astype(float)

    # Previous value for cross detection
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= threshold and current < threshold)
    entries = (prev_rsi >= rsi_oversold) & (rsi < rsi_oversold)

    # Exit: RSI crosses above overbought (prev <= threshold and current > threshold)
    exits = (prev_rsi <= rsi_overbought) & (rsi > rsi_overbought)

    # Build position series: set 1 on entry rows, 0 on exit rows, forward-fill, then fill start as 0
    position = pd.Series(index=close.index, dtype=float)
    position[entries.fillna(False)] = 1.0
    position[exits.fillna(False)] = 0.0

    # Forward-fill positions to maintain exposure until exit; default to flat (0)
    position = position.ffill().fillna(0.0)

    # Ensure integer 0/1 outputs
    position = position.astype(int)

    return {"ohlcv": position}
