import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' mapping to a DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = ohlcv['close']

    # Read parameters with defaults and validation
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    if not (0.0 <= oversold <= 50.0):
        raise ValueError('oversold must be between 0 and 50')
    if not (50.0 <= overbought <= 100.0):
        raise ValueError('overbought must be between 50 and 100')

    # Ensure oversold < overbought
    if oversold >= overbought:
        raise ValueError('oversold threshold must be less than overbought threshold')

    # Calculate RSI using vectorbt's implementation
    rsi_obj = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_obj.rsi

    # Previous RSI (lag 1)
    prev_rsi = pd.Series.shift(rsi, 1)

    # Entry: RSI crosses below oversold (prev >= oversold and current < oversold)
    entry_cond = (pd.Series.fillna(prev_rsi >= oversold, False)) & (pd.Series.fillna(rsi < oversold, False))

    # Exit: RSI crosses above overbought (prev <= overbought and current > overbought)
    exit_cond = (pd.Series.fillna(prev_rsi <= overbought, False)) & (pd.Series.fillna(rsi > overbought, False))

    # Convert conditions to numpy boolean arrays (NaNs handled by fillna above)
    entry_arr = pd.Series.fillna(entry_cond, False).values
    exit_arr = pd.Series.fillna(exit_cond, False).values

    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)

    in_position = False
    for i in range(n):
        # If we have an entry signal and are not in position, enter
        if bool(entry_arr[i]) and not in_position:
            in_position = True
            pos_arr[i] = 1
            continue

        # If we have an exit signal and are in position, exit
        if bool(exit_arr[i]) and in_position:
            in_position = False
            pos_arr[i] = 0
            continue

        # Otherwise, carry forward current state
        if in_position:
            pos_arr[i] = 1
        else:
            pos_arr[i] = 0

    position_series = pd.Series(pos_arr, index=close.index)

    return { 'ohlcv': position_series }
