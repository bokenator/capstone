import pandas as pd
import numpy as np
import vectorbt as vbt
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
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate inputs
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]
    # Validate params and use defaults if missing
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30.0)
    overbought = params.get("overbought", 70.0)

    # Basic validation of param ranges according to PARAM_SCHEMA
    if not (isinstance(rsi_period, int) and 2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be int between 2 and 100")
    if not (isinstance(oversold, (int, float)) and 0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be float between 0 and 50")
    if not (isinstance(overbought, (int, float)) and 50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be float between 50 and 100")

    # Calculate RSI using vectorbt's RSI
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # Prepare numpy arrays for iteration
    rsi_values = rsi.values
    n = len(rsi_values)
    positions = np.zeros(n, dtype=np.int8)  # 0 = flat, 1 = long

    state = 0  # current position state: 0 flat, 1 long

    for i in range(n):
        rv = rsi_values[i]
        prev_rv = rsi_values[i - 1] if i > 0 else np.nan

        # If current RSI is NaN, keep previous state (usually during warmup)
        if np.isnan(rv):
            positions[i] = state
            continue

        # Entry condition: RSI crosses below oversold (prev > oversold and curr <= oversold)
        if state == 0:
            cond_entry = (not np.isnan(prev_rv)) and (prev_rv > oversold) and (rv <= oversold)
            if cond_entry:
                state = 1

        # Exit condition: RSI crosses above overbought (prev < overbought and curr >= overbought)
        else:  # state == 1
            cond_exit = (not np.isnan(prev_rv)) and (prev_rv < overbought) and (rv >= overbought)
            if cond_exit:
                state = 0

        positions[i] = state

    pos_series = pd.Series(positions, index=rsi.index, dtype=np.int8)

    return {"ohlcv": pos_series}
