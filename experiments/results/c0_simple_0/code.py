"""
Reference Implementation: Simple Strategy (RSI Mean Reversion)
==============================================================

Strategy:
- Calculate RSI with configurable period (default 14)
- Go long when RSI crosses below oversold threshold (default 30)
- Exit when RSI crosses above overbought threshold (default 70)
- Long-only, single asset

Function signature:
    generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

Returns position targets in {0, 1} where:
- 1 = long position
- 0 = flat (no position)
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict containing 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters:
            - rsi_period (int): RSI calculation period (default 14)
            - oversold (float): RSI threshold for entry (default 30.0)
            - overbought (float): RSI threshold for exit (default 70.0)

    Returns:
        Dict with 'ohlcv' key mapping to position Series.
        Position values: 1 (long), 0 (flat)
    """
    # Extract data
    ohlcv = data["ohlcv"]
    close = ohlcv["close"]

    # Extract parameters with defaults
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30.0)
    overbought = params.get("overbought", 70.0)

    # Compute RSI using vectorbt
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Initialize position series
    n = len(close)
    position = pd.Series(0.0, index=close.index)

    # Track position state
    in_position = False

    for i in range(n):
        rsi_val = rsi.iloc[i]

        if np.isnan(rsi_val):
            # During warmup period, stay flat
            position.iloc[i] = 0.0
            continue

        if not in_position:
            # Check for entry: RSI below oversold
            if rsi_val < oversold:
                in_position = True
                position.iloc[i] = 1.0
            else:
                position.iloc[i] = 0.0
        else:
            # Check for exit: RSI above overbought
            if rsi_val > overbought:
                in_position = False
                position.iloc[i] = 0.0
            else:
                # Stay in position
                position.iloc[i] = 1.0

    return {"ohlcv": position}
