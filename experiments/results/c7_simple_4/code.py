import typing
from typing import Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


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
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate and extract DataFrame (support both dict input and direct DataFrame)
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data dict must contain 'ohlcv' key with OHLCV DataFrame")
        df = data["ohlcv"]
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("data must be a dict[str, DataFrame] or a DataFrame")

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain 'close' column")

    close = df["close"]

    # Extract params with defaults and basic validation
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    # Calculate RSI using vectorbt (causal)
    # Use fully-qualified API call per VAS requirements
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Prepare numpy arrays for fast, explicit loop without lookahead
    rsi_vals = pd.Series.fillna(rsi, np.nan).values  # ensure numpy array (fillna used as qualified call)
    prev_vals = pd.Series.shift(rsi, 1).values

    n = len(rsi_vals)
    positions = np.zeros(n, dtype=np.int8)  # 0 = flat, 1 = long

    # Compute entry/exit boolean arrays (element-wise, uses only past and current values)
    entry_arr = (prev_vals >= oversold) & (rsi_vals < oversold)
    exit_arr = (prev_vals <= overbought) & (rsi_vals > overbought)

    # Iterate forward in time to avoid double entries and ensure no lookahead
    in_position = False
    for i in range(n):
        # Skip if RSI is not finite (warmup)
        if not np.isfinite(rsi_vals[i]):
            positions[i] = 0
            continue

        if not in_position:
            # Only enter when condition met and not already in position
            if entry_arr[i]:
                in_position = True
                positions[i] = 1
            else:
                positions[i] = 0
        else:
            # Currently long: check for exit
            if exit_arr[i]:
                in_position = False
                positions[i] = 0
            else:
                positions[i] = 1

    pos_series = pd.Series(positions, index=close.index)

    return {"ohlcv": pos_series}
