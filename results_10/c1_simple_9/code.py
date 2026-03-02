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
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain an 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close: pd.Series = df["close"].astype("float64")

    # Validate and extract parameters (only allowed params are accessed)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Use provided params or defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation of parameter ranges according to PARAM_SCHEMA
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    def compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)

        # Use EWMA with alpha=1/period and require at least `period` observations
        # to produce a reliable value (min_periods=period).
        ma_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        ma_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        rs = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    rsi = compute_rsi(close, rsi_period)

    # Define crossing conditions
    # Entry: RSI crosses below the oversold level (from >= oversold to < oversold)
    # Exit: RSI crosses above the overbought level (from <= overbought to > overbought)
    prev_rsi = rsi.shift(1)

    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Initialize position series (0 = flat, 1 = long)
    n = len(close)
    entries = entry_mask.to_numpy(dtype=bool)
    exits = exit_mask.to_numpy(dtype=bool)

    pos = np.zeros(n, dtype=np.int8)
    current = 0

    # State machine to build target position for each bar
    for i in range(n):
        if current == 1 and exits[i]:
            # Exit on this bar
            current = 0
        elif current == 0 and entries[i]:
            # Enter on this bar
            current = 1
        pos[i] = current

    position_series = pd.Series(pos, index=close.index, name="position")

    return {"ohlcv": position_series}
