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
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' with 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series. Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = ohlcv["close"]
    if not isinstance(close, pd.Series):
        # Ensure close is a Series
        close = pd.Series(close)

    n = len(close)

    # Extract and validate params with defaults
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using vectorbt (uses past data only)
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # Previous RSI (lag 1)
    prev_rsi = pd.Series.shift(rsi, 1)

    # Entry when RSI crosses below oversold: prev > oversold and curr <= oversold
    entry_signal = (prev_rsi > oversold) & (rsi <= oversold)
    entry_signal = pd.Series.fillna(entry_signal, False)

    # Exit when RSI crosses above overbought: prev < overbought and curr >= overbought
    exit_signal = (prev_rsi < overbought) & (rsi >= overbought)
    exit_signal = pd.Series.fillna(exit_signal, False)

    # Build position series iteratively to avoid double entries and ensure no lookahead
    pos = np.zeros(n, dtype=np.int8)
    entry_arr = entry_signal.values
    exit_arr = exit_signal.values

    in_pos = False
    for i in range(n):
        if i > 0:
            pos[i] = pos[i - 1]
        else:
            pos[i] = 0

        # If exit signal and currently in position -> go flat
        if exit_arr[i] and in_pos:
            pos[i] = 0
            in_pos = False
            continue

        # If entry signal and not in position -> go long
        if entry_arr[i] and not in_pos:
            pos[i] = 1
            in_pos = True
            continue

        # Otherwise keep previous state
        # (already assigned above)

    position_series = pd.Series(pos, index=close.index, dtype=np.int8)

    return {"ohlcv": position_series}
