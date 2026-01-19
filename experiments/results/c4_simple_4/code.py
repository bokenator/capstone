import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate input data
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close']

    # Read and validate params with defaults from PARAM_SCHEMA
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Basic sanity checks
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    if not (0.0 <= oversold <= 50.0):
        raise ValueError('oversold must be between 0 and 50')
    if not (50.0 <= overbought <= 100.0):
        raise ValueError('overbought must be between 50 and 100')

    n = len(close)
    # Prepare default zero position series
    positions = pd.Series(np.zeros(n, dtype=np.int8), index=close.index)

    # If not enough data, return zeros
    if n == 0:
        return {'ohlcv': positions}

    # Compute RSI using vectorbt's RSI (VAS-approved)
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Previous RSI (use module-qualified call per VAS)
    prev_rsi = pd.Series.shift(rsi, 1)

    # Entry: RSI crosses below oversold (prev > oversold and curr <= oversold)
    entry_mask = (prev_rsi > oversold) & (rsi <= oversold)

    # Exit: RSI crosses above overbought (prev < overbought and curr >= overbought)
    exit_mask = (prev_rsi < overbought) & (rsi >= overbought)

    # Ensure masks are aligned and fill NA with False
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series: long-only. Enter on entry_mask, exit on exit_mask.
    pos_arr = np.zeros(n, dtype=np.int8)
    in_long = False
    for i in range(n):
        # Skip if RSI is NaN (warmup)
        if not np.isfinite(rsi.values[i]):
            pos_arr[i] = 0
            continue

        if in_long:
            # If currently long, check for exit
            if exit_mask.values[i]:
                in_long = False
                pos_arr[i] = 0
            else:
                pos_arr[i] = 1
        else:
            # If not long, check for entry
            if entry_mask.values[i]:
                in_long = True
                pos_arr[i] = 1
            else:
                pos_arr[i] = 0

    positions = pd.Series(pos_arr, index=close.index)

    return {'ohlcv': positions}
