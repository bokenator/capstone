"""
RSI Mean Reversion Signal Generator

Implements generate_signals as specified in the prompt.
- Calculates RSI using Wilder's smoothing (EMA with alpha=1/period)
- Generates long entries when RSI crosses below oversold threshold
- Generates exits when RSI crosses above overbought threshold
- Long-only, single asset (ohlcv slot)

The function accepts the 'data' dict with 'ohlcv' DataFrame containing 'close' column.
For convenience it also accepts a DataFrame directly (treated as the 'ohlcv' slot),
but the function signature remains unchanged.

Returns a dict: {"ohlcv": pd.Series} of positions (+1 long, 0 flat, -1 short)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
              For convenience, a DataFrame may be passed directly and will be
              treated as {'ohlcv': data}.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate and normalize inputs
    if data is None:
        raise ValueError("`data` must be provided and contain 'ohlcv' DataFrame with 'close' column")

    # Allow passing a DataFrame directly for convenience
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("`data` dict must contain 'ohlcv' DataFrame")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("`data` must be a pandas DataFrame or a dict with 'ohlcv' DataFrame")

    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("'ohlcv' must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    # Extract close prices
    close = ohlcv["close"].astype(float).copy()

    # Validate params and set defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Calculate RSI using Wilder's smoothing (EMA with alpha=1/period)
    # This implementation uses only past data (no lookahead)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use ewm with adjust=False to match Wilder's smoothing
    # alpha = 1 / period
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss is zero -> RSI = 100; where avg_gain is zero -> RSI = 0
    rsi = rsi.fillna(0.0)
    rsi[(avg_loss == 0.0) & (avg_gain > 0.0)] = 100.0

    # Ensure RSI index matches close index
    rsi = rsi.reindex(close.index)

    # Generate raw crossing signals (vectorized)
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and curr < oversold
    entries_raw = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit when RSI crosses above overbought: prev <= overbought and curr > overbought
    exits_raw = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series ensuring no double entries and long-only behavior
    position = pd.Series(0, index=close.index, dtype="int8")
    in_position = False

    # Iterate once left-to-right ensuring no future dependence
    for ts in range(len(position)):
        if ts == 0:
            position.iloc[ts] = 0
            continue

        if in_position:
            # If currently long, check for exit
            if exits_raw.iloc[ts]:
                in_position = False
                position.iloc[ts] = 0
            else:
                # remain long
                position.iloc[ts] = 1
        else:
            # If flat, check for entry
            if entries_raw.iloc[ts]:
                in_position = True
                position.iloc[ts] = 1
            else:
                position.iloc[ts] = 0

    # Ensure no NaNs in position (shouldn't be any)
    position = position.fillna(0).astype(int)

    # Return mapping
    return {"ohlcv": position}
