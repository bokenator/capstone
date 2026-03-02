# -*- coding: utf-8 -*-
"""
RSI Mean Reversion Signal Generator

Implements generate_signals(...) as specified in the prompt.

- Calculates RSI using Wilder's smoothing (EWMA with alpha=1/period)
- Goes long when RSI crosses below the oversold threshold
- Exits when RSI crosses above the overbought threshold
- Long-only, single asset ("ohlcv" slot)

The function is careful to avoid lookahead bias (only uses past data
for RSI calculation and crossing detection), handles NaNs and warmup,
and returns a position Series with values 0 or 1.
"""
from typing import Dict

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict,
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series (values 0 or 1).
    """

    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot name to DataFrame")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] DataFrame must contain 'close' column")

    close: pd.Series = df["close"].astype(float).copy()

    # Ensure index is preserved and is monotonic; work with a copy to avoid side effects
    index = close.index

    # Extract parameters with validation against PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    oversold = float(params.get("oversold", 30.0))
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")

    overbought = float(params.get("overbought", 70.0))
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")

    # Helper: compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    # This implementation is causal (uses only past data).
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        # series: close prices
        delta = series.diff()

        # Gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Use Wilder's smoothing via ewm (alpha = 1/period), min_periods=period to
        # avoid unstable early values. This is causal (adjust=False).
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # rsi will be NaN until enough data (min_periods). Keep NaNs as-is.
        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings without lookahead
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (from >= oversold to < oversold)
    entry = (prev_rsi >= oversold) & (rsi < oversold)
    # If previous RSI is NaN but current RSI is below oversold (first valid RSI), consider it an entry
    entry |= (prev_rsi.isna() & rsi.notna() & (rsi < oversold))

    # Exit: RSI crosses above overbought (from <= overbought to > overbought)
    exit = (prev_rsi <= overbought) & (rsi > overbought)
    # Do NOT treat first valid RSI > overbought as an exit since we would not be in a position yet

    # Replace NaN booleans with False for safety
    entry = entry.fillna(False)
    exit = exit.fillna(False)

    # Build position series iteratively to ensure no double entries/exits and long-only behavior
    pos_values = np.zeros(len(close), dtype="int8")
    in_position = 0

    # Iterate by positional index to avoid lookahead and preserve determinism
    for i in range(len(close)):
        if entry.iat[i] and in_position == 0:
            in_position = 1
        elif exit.iat[i] and in_position == 1:
            in_position = 0
        pos_values[i] = in_position

    position = pd.Series(pos_values, index=index, name="position").astype(int)

    # Ensure only 0 or 1 values
    position = position.where(position.isin([0, 1]), 0).astype(int)

    # Replace any remaining NaNs (should not be any) with 0
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
