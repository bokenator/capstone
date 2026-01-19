from typing import Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict, params: Dict) -> Dict[str, pd.Series]:
    """
    Generate long-only position series using RSI mean reversion strategy.

    Logic:
    - Compute RSI with period 14
    - Entry when RSI crosses below 30 (prev >= 30 and current < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and current > 70)

    Args:
        data: Dict containing 'ohlcv' -> DataFrame with 'close' series
        params: Dict of parameters (not used, kept for compatibility)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (1 for long, 0 for flat)
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv must contain 'close' column/series")

    close = ohlcv["close"]

    # Compute RSI using vectorbt (window fixed at 14 as per spec)
    rsi_series = vbt.RSI.run(close, window=14).rsi

    # Convert to numpy arrays for deterministic, no-lookahead iterative logic
    rsi_arr = rsi_series.values
    prev_rsi_arr = pd.Series.shift(rsi_series, 1).values

    n = len(rsi_arr)
    entries = np.zeros(n, dtype=np.bool_)
    exits = np.zeros(n, dtype=np.bool_)

    # Determine cross conditions (no future data used)
    # Entry: prev >= 30 and current < 30
    # Exit: prev <= 70 and current > 70
    for i in range(1, n):
        prev = prev_rsi_arr[i]
        curr = rsi_arr[i]
        # Skip if any is NaN
        if np.isnan(prev) or np.isnan(curr):
            continue
        if (prev >= 30.0) and (curr < 30.0):
            entries[i] = True
        if (prev <= 70.0) and (curr > 70.0):
            exits[i] = True

    # Build position series ensuring no double entries (stateful, sequential)
    positions = np.zeros(n, dtype=np.int64)
    in_position = False
    for i in range(n):
        if entries[i] and not in_position:
            in_position = True
            positions[i] = 1
            continue
        if exits[i] and in_position:
            in_position = False
            positions[i] = 0
            continue
        # maintain previous state
        positions[i] = 1 if in_position else 0

    position_series = pd.Series(np.array(positions, dtype=np.int64), index=close.index)

    return {"ohlcv": position_series}
