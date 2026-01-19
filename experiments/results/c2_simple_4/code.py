import typing
import pandas as pd
import numpy as np
import vectorbt as vbt


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy:
    - RSI period = 14
    - Enter long when RSI crosses below 30
    - Exit long when RSI crosses above 70
    - Long-only, single asset

    Args:
        data: Dict containing 'ohlcv' DataFrame with a 'close' column.
        params: Dict of parameters (not used but kept for compatibility).

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of position targets
        (+1 for long, 0 for flat), indexed like the input close prices.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data['ohlcv']
    if 'close' not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close']
    # Compute RSI using vectorbt's implementation (window fixed at 14)
    rsi = vbt.RSI.run(close, window=14).rsi

    # Previous RSI (shifted by 1). Use module-qualified pandas API per VAS.
    prev_rsi = pd.Series.shift(rsi, 1)

    # Entry: prev_rsi >= 30 and rsi < 30 (cross below 30)
    entries = (pd.Series.fillna(prev_rsi, np.inf) >= 30) & (pd.Series.fillna(rsi, np.inf) < 30)
    # Exit: prev_rsi <= 70 and rsi > 70 (cross above 70)
    exits = (pd.Series.fillna(prev_rsi, -np.inf) <= 70) & (pd.Series.fillna(rsi, -np.inf) > 70)

    # Initialize position series (0 = flat, 1 = long)
    pos_values = np.zeros(len(close), dtype=np.int8)

    # Convert boolean series to numpy arrays for faster iteration and safe nan handling
    entries_arr = entries.fillna(False).values
    exits_arr = exits.fillna(False).values

    # Iterate and build positions (simple state machine)
    for i in range(len(pos_values)):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = pos_values[i - 1]

        if prev_pos == 0:
            # If flat, enter on entry signal
            if entries_arr[i]:
                pos_values[i] = 1
            else:
                pos_values[i] = 0
        else:
            # If long, exit on exit signal
            if exits_arr[i]:
                pos_values[i] = 0
            else:
                pos_values[i] = 1

    positions = pd.Series(pos_values, index=close.index, dtype=np.int8)

    return {"ohlcv": positions}
