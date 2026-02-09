"""
RSI Mean Reversion Signal Generator

Implements a simple long-only mean reversion strategy based on RSI:
- RSI period = 14 (configurable via params['rsi_window'])
- Go long when RSI crosses below 30
- Exit long when RSI crosses above 70

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

The returned dict must contain the key 'ohlcv' mapping to a pd.Series of position
targets: 1 = long, 0 = flat. The Series index must match data['ohlcv']['close'] index.

This module depends on pandas, numpy, and vectorbt.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing market data. Must include key 'ohlcv' with a
              DataFrame that contains a 'close' column (pd.Series).
        params: Dictionary of parameters. Supported keys:
            - 'rsi_window' (int): RSI period/window. Defaults to 14.

    Returns:
        A dictionary with a single key 'ohlcv' whose value is a pd.Series of
        position targets (1 for long, 0 for flat). The Series index matches
        the input close series index.

    Raises:
        ValueError: If required data/columns are missing or multiple assets are provided.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data.get("ohlcv")
    if ohlcv is None:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain a 'close' column")

    close = ohlcv["close"]

    # If close is a DataFrame, only support single-column close
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("generate_signals supports single-asset data only")

    if not isinstance(close, pd.Series):
        # Try to coerce to Series
        close = pd.Series(close)

    # Parameters
    rsi_window = int(params.get("rsi_window", 14)) if params is not None else 14

    # Compute RSI using vectorbt's RSI indicator (verified: vbt.RSI.run(...).rsi)
    rsi_res = vbt.RSI.run(close, window=rsi_window)
    # The result exposes .rsi attribute
    if hasattr(rsi_res, "rsi"):
        rsi = rsi_res.rsi
    else:
        # Fallback: sometimes the indicator might return a Series directly
        rsi = rsi_res

    # If rsi is a DataFrame with single column, convert to Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            raise ValueError("RSI returned multiple columns; only single-asset supported")

    # Ensure alignment
    rsi = rsi.reindex(close.index)

    # Determine cross signals. Use previous-bar comparison and require both current
    # and previous RSI to be non-NaN to avoid spurious triggers during warmup.
    rsi_prev = rsi.shift(1)
    valid = rsi.notna() & rsi_prev.notna()

    # Entry when RSI crosses below 30: previous >= 30 and current < 30
    entries = valid & (rsi_prev >= 30) & (rsi < 30)

    # Exit when RSI crosses above 70: previous <= 70 and current > 70
    exits = valid & (rsi_prev <= 70) & (rsi > 70)

    # Build position series: 1 when in long, 0 when flat.
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate through chronological index
    for i in range(len(close.index)):
        if entries.iloc[i] and not in_position:
            in_position = True
            position.iloc[i] = 1
        elif exits.iloc[i] and in_position:
            in_position = False
            position.iloc[i] = 0
        else:
            position.iloc[i] = 1 if in_position else 0

    return {"ohlcv": position}
