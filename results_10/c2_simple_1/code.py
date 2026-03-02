"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean reversion strategy:
- RSI period = 14
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)
- Long-only, single asset

Exports:
- generate_signals(data, params) -> dict[str, pd.Series]

The function expects `data` to be a dict with key 'ohlcv' containing a DataFrame
with a 'close' column. It returns a dict with key 'ohlcv' mapping to a Series
of position targets (1 = long, 0 = flat) aligned with the input index.

This code uses vectorbt's RSI implementation to compute RSI and follows
vectorbt's IndicatorFactory.run signature: vbt.RSI.run(close, window=14)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv' with a
            DataFrame that has a 'close' column.
        params: Parameters dictionary (not used by this simple implementation,
            kept for compatibility with the backtest runner).

    Returns:
        A dictionary with key 'ohlcv' mapping to a pandas Series of integer
        positions (1 = long, 0 = flat), aligned to the input index.

    Raises:
        ValueError: If required data is missing or malformed.
    """

    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Compute RSI using vectorbt's RSI indicator (window=14)
    # Verified with vectorbt docs: vbt.RSI.run(close, window=14) -> result with .rsi
    rsi_res = vbt.RSI.run(close, window=14)
    rsi = rsi_res.rsi

    # Ensure alignment and series type
    if isinstance(rsi, pd.DataFrame):
        # If vectorbt returned a DataFrame (e.g., multi-column input), try to
        # reduce to a single series that matches our close column
        if close.name in rsi.columns:
            rsi = rsi[close.name]
        else:
            # Take the first column
            rsi = rsi.iloc[:, 0]

    rsi = rsi.reindex(close.index)

    # Create entry/exit signals based on RSI crossings
    # Entry: RSI crosses below 30 -> previous >= 30 and current < 30
    # Exit: RSI crosses above 70 -> previous <= 70 and current > 70
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi >= 30) & (rsi < 30)
    exit_signal = (prev_rsi <= 70) & (rsi > 70)

    # Replace NaN comparisons with False to avoid spurious signals during warmup
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series (0 or 1), forward filling after entries until exits
    positions = pd.Series(0, index=close.index, dtype=int)

    # Iterate through time to maintain a single long position at a time
    in_position = False
    for idx in range(len(close)):
        if entry_signal.iloc[idx] and not in_position:
            in_position = True
        elif exit_signal.iloc[idx] and in_position:
            in_position = False
        positions.iloc[idx] = 1 if in_position else 0

    # Ensure no NaNs and proper dtype
    positions = positions.fillna(0).astype(int)

    return {'ohlcv': positions}
