import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period
            - oversold (float): RSI threshold for entry (go long)
            - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series. Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data['ohlcv']
    if 'close' not in df.columns:
        raise KeyError("'ohlcv' DataFrame must contain 'close' column")

    # Extract and validate parameters (use only declared params)
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    close = df['close'].astype(float).copy()

    # Handle empty input
    if close.empty:
        return {'ohlcv': pd.Series(dtype='int64')}

    # Calculate RSI using Wilder's smoothing (EWMA with alpha = 1/period, adjust=False)
    delta = close.diff()
    # For first delta, set to 0 to avoid NaNs affecting EWM
    delta.iloc[0] = 0.0

    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Use ewm with adjust=False for Wilder's smoothing (causal)
    alpha = 1.0 / float(rsi_period)
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

    # Compute RSI safely
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle cases where avg_loss == 0 -> RSI = 100, and avg_gain == 0 -> RSI = 0
    rsi = rsi.fillna(0.0)
    rsi[avg_loss == 0.0] = 100.0

    # Define crossing conditions (use only past and current values, no future data)
    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series: long-only, single asset. Values: 1 (long) or 0 (flat)
    position = pd.Series(0, index=close.index, dtype='int8')

    in_position = False
    # Iterate sequentially to avoid double entries and ensure causality
    for i in range(len(close)):
        if entry_cond.iloc[i] and not in_position:
            in_position = True
        elif exit_cond.iloc[i] and in_position:
            in_position = False
        position.iloc[i] = 1 if in_position else 0

    # Return positions mapped to slots. No NaNs in output.
    return {'ohlcv': position}
