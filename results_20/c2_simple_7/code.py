"""
RSI Mean Reversion Signal Generator

Implements a simple long-only RSI mean reversion strategy:
- RSI period: configurable (default 14)
- Go long when RSI crosses below oversold threshold (default 30)
- Exit long when RSI crosses above overbought threshold (default 70)

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

Returns a dict with key 'ohlcv' mapping to a pd.Series of position targets:
- 1 for long, 0 for flat

This module uses vectorbt's RSI indicator for computation. The function is
robust to NaNs/warmup periods and validates inputs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv' with a
            pandas.DataFrame that has a 'close' column.
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'rsi_oversold' (float): Oversold threshold. Default 30.
            - 'rsi_overbought' (float): Overbought threshold. Default 70.

    Returns:
        Dict with key 'ohlcv' containing a pandas.Series of position targets
        (1 = long, 0 = flat), indexed the same as the input close series.

    Raises:
        ValueError: If required data is missing or malformed.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")

    if 'ohlcv' not in data:
        raise ValueError("data dict must contain 'ohlcv' key with a DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close']
    if isinstance(close, pd.DataFrame):
        # If close is a single-column DataFrame, convert to Series
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("close must be a single series (1D)")

    # Parameters with sensible defaults
    rsi_period = int(params.get('rsi_period', 14))
    rsi_oversold = float(params.get('rsi_oversold', 30.0))
    rsi_overbought = float(params.get('rsi_overbought', 70.0))

    # Compute RSI using vectorbt's indicator
    # Signature verified: vbt.RSI.run(close, window=14, ewm=False)
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_res.rsi

    # Squeeze to Series if needed
    if isinstance(rsi, pd.DataFrame):
        # If RSI returns 1 column, take that column
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            # If multiple columns are present (unexpected), try to squeeze
            rsi = rsi.squeeze()

    # Ensure index alignment with close
    rsi = rsi.reindex(close.index)

    # Build crossing signals. Use previous value to detect crosses.
    prev = rsi.shift(1)

    entry_signals = (prev >= rsi_oversold) & (rsi < rsi_oversold)
    exit_signals = (prev <= rsi_overbought) & (rsi > rsi_overbought)

    # Replace NaNs with False: no signals during warmup
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Generate position series (1 = long, 0 = flat)
    n = len(close)
    position = np.zeros(n, dtype=int)

    in_position = False
    for i, idx in enumerate(close.index):
        if not in_position:
            if bool(entry_signals.iloc[i]):
                in_position = True
                position[i] = 1
            else:
                position[i] = 0
        else:
            # Currently in a position
            if bool(exit_signals.iloc[i]):
                # Exit at this bar
                in_position = False
                position[i] = 0
            else:
                # Stay long
                position[i] = 1

    position_series = pd.Series(position, index=close.index, name='position')

    return {"ohlcv": position_series}
