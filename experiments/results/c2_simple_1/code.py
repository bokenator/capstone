from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for a simple RSI mean-reversion strategy.

    Strategy:
    - RSI period = 14
    - Go long when RSI crosses below 30
    - Exit when RSI crosses above 70
    - Long-only, single asset

    Args:
        data: Dict containing price dataframes. Expects data['ohlcv'] with a 'close' column.
        params: Dict of parameters (not used for this simple strategy but accepted for signature compatibility).

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (+1 for long, 0 for flat).
    """

    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"]

    # Calculate RSI using vectorbt
    rsi: pd.Series = vbt.RSI.run(close, window=14).rsi

    # Previous RSI (shifted by 1)
    prev_rsi: pd.Series = pd.Series.shift(rsi, 1)

    # Entry: prev_rsi >= 30 and current rsi < 30  (cross below 30)
    prev_ge_30: pd.Series = pd.Series.fillna(prev_rsi >= 30, False)
    entry_series: pd.Series = prev_ge_30 & (rsi < 30)

    # Exit: prev_rsi <= 70 and current rsi > 70 (cross above 70)
    prev_le_70: pd.Series = pd.Series.fillna(prev_rsi <= 70, False)
    exit_series: pd.Series = prev_le_70 & (rsi > 70)

    # Convert boolean series to numpy arrays for iteration
    entries: np.ndarray = np.array(entry_series, dtype=bool)
    exits: np.ndarray = np.array(exit_series, dtype=bool)

    n = len(close)
    positions = np.zeros(n, dtype=np.int8)

    # Scan through bars to build position: long after entry until exit
    in_pos = False
    for i in range(n):
        if entries[i] and not in_pos:
            in_pos = True
            positions[i] = 1
        elif in_pos:
            # If currently in position, remain in unless an exit occurs
            if exits[i]:
                in_pos = False
                positions[i] = 0
            else:
                positions[i] = 1
        else:
            positions[i] = 0

    positions_series: pd.Series = pd.Series(positions, index=close.index, dtype=np.int8)

    return {"ohlcv": positions_series}
