from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position targets (+1 long, 0 flat) based on RSI mean reversion strategy.

    Strategy logic:
    - RSI period = 14
    - Go long when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and curr > 70)
    - Long-only, single asset

    Args:
        data: Dictionary of dataframes; expects data['ohlcv'] with a 'close' column (pd.Series)
        params: Parameter dict (not used for now, kept for compatibility)

    Returns:
        A dict with key 'ohlcv' mapping to a pd.Series of position targets (+1 or 0), indexed like the input close series.
    """

    # Validate inputs
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"]

    # Calculate RSI using vectorbt (period 14)
    rsi_series: pd.Series = vbt.RSI.run(close, window=14).rsi

    # Previous RSI (shifted by 1 bar)
    prev_rsi: pd.Series = pd.Series.shift(rsi_series, 1)

    # Entry when RSI crosses below 30: prev >= 30 and curr < 30
    raw_entries = (prev_rsi >= 30) & (rsi_series < 30)
    entries = pd.Series.fillna(raw_entries, False)

    # Exit when RSI crosses above 70: prev <= 70 and curr > 70
    raw_exits = (prev_rsi <= 70) & (rsi_series > 70)
    exits = pd.Series.fillna(raw_exits, False)

    # Convert masks to numpy arrays for fast iteration and to avoid using unlisted pandas APIs
    entries_arr = entries.values
    exits_arr = exits.values

    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)

    in_position = False
    for i in range(n):
        # Skip bars where RSI is NaN (warmup) - remain flat
        # Use the RSI series' values to check for NaN
        if np.isnan(rsi_series.values[i]):
            pos_arr[i] = 1 if in_position else 0
            continue

        if (not in_position) and entries_arr[i]:
            in_position = True
            pos_arr[i] = 1
        elif in_position and exits_arr[i]:
            in_position = False
            pos_arr[i] = 0
        else:
            pos_arr[i] = 1 if in_position else 0

    # Construct position series with the same index as the input close series
    position_series = pd.Series(pos_arr, index=close.index, dtype=int)

    return {"ohlcv": position_series}
