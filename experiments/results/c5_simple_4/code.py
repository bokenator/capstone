import pandas as pd
import numpy as np
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
              Alternatively, a single DataFrame can be passed in place of the dict;
              in that case it is interpreted as the 'ohlcv' DataFrame.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate and normalize input (allow passing DataFrame directly)
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if 'ohlcv' not in data:
            raise ValueError("data dict must contain 'ohlcv' key")
        ohlcv = data['ohlcv']
    else:
        raise TypeError("data must be a pandas DataFrame or a dict containing 'ohlcv' DataFrame")

    # Ensure required column exists
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    # Extract parameters with validation
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    close = ohlcv['close'].astype(float)

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    entry_cond = entry_cond.fillna(False)
    exit_cond = exit_cond.fillna(False)

    position = pd.Series(0, index=close.index, dtype='int8')

    in_position = False
    for idx in range(len(close)):
        if in_position:
            if exit_cond.iloc[idx]:
                in_position = False
                position.iloc[idx] = 0
            else:
                position.iloc[idx] = 1
        else:
            if entry_cond.iloc[idx]:
                in_position = True
                position.iloc[idx] = 1
            else:
                position.iloc[idx] = 0

    position = position.fillna(0).astype(int)

    return {'ohlcv': position}
