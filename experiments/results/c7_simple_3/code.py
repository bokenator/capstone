import numpy as np
import pandas as pd
from typing import Dict
import vectorbt as vbt


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
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]
    # Validate params and apply defaults
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2:
        rsi_period = 2
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using vectorbt (no lookahead)
    rsi_obj = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_obj.rsi  # pd.Series

    n = len(close)
    rsi_values = rsi.values

    # Prepare previous RSI values (no future data)
    prev_rsi = np.empty_like(rsi_values, dtype=np.float64)
    if n > 0:
        prev_rsi[0] = np.nan
    if n > 1:
        prev_rsi[1:] = rsi_values[:-1]

    # Initialize positions array: 0 = flat, 1 = long
    positions = np.zeros(n, dtype=np.int8)
    in_position = False

    # Iterate through bars to avoid double entries and ensure causality
    for i in range(n):
        cur_rsi = rsi_values[i]
        p_rsi = prev_rsi[i]

        # If RSI is NaN (warmup), carry previous position (likely 0)
        if np.isnan(cur_rsi):
            if i > 0:
                positions[i] = positions[i - 1]
            else:
                positions[i] = 0
            in_position = bool(positions[i] == 1)
            continue

        # Entry condition: previous RSI >= oversold and current RSI < oversold
        entry = False
        if (not in_position) and (not np.isnan(p_rsi)) and (p_rsi >= oversold) and (cur_rsi < oversold):
            entry = True

        # Exit condition: previous RSI <= overbought and current RSI > overbought
        exit_sig = False
        if in_position and (not np.isnan(p_rsi)) and (p_rsi <= overbought) and (cur_rsi > overbought):
            exit_sig = True

        if entry:
            positions[i] = 1
            in_position = True
        elif exit_sig:
            positions[i] = 0
            in_position = False
        else:
            # Carry forward previous position
            if i > 0:
                positions[i] = positions[i - 1]
            else:
                positions[i] = 0

    # Build pandas Series for output with same index as input
    position_series = pd.Series(positions, index=close.index)

    return {"ohlcv": position_series}
