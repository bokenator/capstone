import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
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
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = ohlcv['close']
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Validate params and use defaults from schema if missing
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Calculate RSI using vectorbt
    # Use fully-qualified call as required by VAS
    rsi_obj = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_obj.rsi

    # Prepare shifted RSI for cross detection using module-qualified pandas calls
    prev_rsi = pd.Series.shift(rsi, 1)

    # Entry: RSI crosses below oversold (prev >= oversold and current < oversold)
    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and current > overbought)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    # Initialize positions array: 0 = flat, 1 = long
    positions = np.zeros(len(rsi), dtype=np.int8)

    state = 0  # current position state: 0 flat, 1 long
    rsi_values = rsi.values
    entry_vals = entry_cond.fillna(False).values
    exit_vals = exit_cond.fillna(False).values

    for i in range(len(rsi_values)):
        # If RSI is NaN (warmup), remain flat
        if np.isnan(rsi_values[i]):
            state = 0
            positions[i] = 0
            continue

        if state == 0:
            if entry_vals[i]:
                state = 1
        elif state == 1:
            if exit_vals[i]:
                state = 0

        positions[i] = state

    pos_series = pd.Series(positions, index=rsi.index, dtype=np.int8)

    return {'ohlcv': pos_series}
