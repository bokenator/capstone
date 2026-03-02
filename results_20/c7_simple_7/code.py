# RSI mean reversion signal generator
from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Position values are: 1 (long) or 0 (flat).
    """
    # Validate input type
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames (expected 'ohlcv')")

    if "ohlcv" not in data:
        raise KeyError("data must contain the key 'ohlcv' with a DataFrame containing a 'close' column")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas.DataFrame")

    if "close" not in df.columns:
        raise KeyError("Input DataFrame must contain 'close' column")

    close = df["close"].astype(float).copy()

    # Read and validate params (use defaults if missing)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Parameter validation according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    # Compute RSI using Wilder's smoothing (exponential smoothing with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential moving average with alpha=1/period and adjust=False for Wilder's smoothing
    alpha = 1.0 / float(rsi_period)
    # min_periods ensures values are NaN until we have rsi_period observations (consistent with RSI warmup)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()

    # Relative Strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss is zero
    rsi = rsi.copy()
    rsi[avg_loss == 0] = 100.0
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0

    # Signals: detect cross below oversold for entry and cross above overbought for exit.
    # Use previous RSI (lag 1). For first valid RSI value where previous is NaN, treat previous as being on the opposite side
    # so that if RSI starts already below oversold we still generate an entry at that first bar.
    prev_rsi_for_entry = rsi.shift(1).fillna(oversold + 1.0)
    prev_rsi_for_exit = rsi.shift(1).fillna(overbought - 1.0)

    entry_signal = (prev_rsi_for_entry >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi_for_exit <= overbought) & (rsi > overbought)

    # Build position series: 1 when long, 0 when flat. Long-only, single asset.
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = False
    # Iterate to avoid double entries/exits and ensure causal behavior
    for i in range(len(position)):
        if not in_position:
            if bool(entry_signal.iloc[i]):
                in_position = True
                position.iloc[i] = 1
            else:
                position.iloc[i] = 0
        else:
            # currently long
            if bool(exit_signal.iloc[i]):
                in_position = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1

    # Ensure no NaNs exist in the returned position series
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
