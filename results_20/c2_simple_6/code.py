"""
RSI Mean Reversion Signal Generator

Implements:
- RSI period 14
- Long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long-only, single asset

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

The returned dictionary must contain the key 'ohlcv' mapping to a pd.Series of
position targets: 1 = long, 0 = flat.

This file is intended to be used with the provided backtest runner.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean-reversion strategy.

    Args:
        data: Dictionary containing market data. Expected to have key 'ohlcv'
            mapping to a DataFrame with a 'close' column.
        params: Dictionary of parameters (not required for this fixed strategy).

    Returns:
        A dictionary with key 'ohlcv' and value being a pd.Series of positions
        indexed the same as the input close prices. Values are 1 for long and 0
        for flat.

    Raises:
        ValueError: If required data is missing or malformed.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' mapping to a DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close']
    # Ensure close is a Series
    if isinstance(close, pd.DataFrame):
        # If it's a single-column DataFrame, convert to Series
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("close must be a pandas Series or a single-column DataFrame")

    # Parameters (fixed for this strategy)
    rsi_period = 14
    oversold = 30.0
    overbought = 70.0

    # Compute RSI using vectorbt's indicator
    # vbt.RSI.run returns an object with `.rsi` output
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    # rsi_res.rsi is a Series (or DataFrame for multiple columns)
    rsi = rsi_res.rsi

    # Work with a 1-D Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            raise ValueError("RSI returned multiple columns; expected single series")

    # Align indexes
    rsi = rsi.reindex(close.index)

    # Identify crossings:
    # Entry: RSI crosses below oversold (prev >= oversold AND curr < oversold)
    # Exit: RSI crosses above overbought (prev <= overbought AND curr > overbought)
    prev = rsi.shift(1)

    entry_mask = (prev >= oversold) & (rsi < oversold) & (~prev.isna()) & (~rsi.isna())
    exit_mask = (prev <= overbought) & (rsi > overbought) & (~prev.isna()) & (~rsi.isna())

    # Build marker series: 1 for entry, 0 for exit, NaN otherwise
    markers = pd.Series(data=np.nan, index=rsi.index, dtype=float)
    markers.loc[exit_mask] = 0.0
    markers.loc[entry_mask] = 1.0

    # Forward-fill markers to create a stateful position series
    # Fill initial NaNs with 0 (start flat)
    position = markers.ffill().fillna(0.0).astype(int)

    # Ensure the returned series has the same name and index as input
    position.name = 'position'

    return {'ohlcv': position}
