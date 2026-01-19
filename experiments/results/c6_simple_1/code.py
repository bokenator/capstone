"""
RSI Mean Reversion Signal Generator

Implements:
- RSI period = 14 (vbt.RSI.run)
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long-only, single asset

Exported function:
- generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

Notes on API usage:
- Uses only APIs from the Verified API Surface (VAS).
- Pandas Series methods are invoked via the pd.Series.<method>(series, ...) form
  to comply with the VAS requirement of fully-qualified calls.

"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position series for RSI mean reversion strategy.

    Args:
        data: Dictionary containing OHLCV DataFrame under key "ohlcv". The DataFrame
            must contain a "close" column as a pandas Series.
        params: Parameters dictionary (not used for now, present for compatibility).

    Returns:
        A dictionary with key "ohlcv" mapping to a pandas Series of positions:
        +1 for long, 0 for flat. The Series index matches the input close Series.

    Behavior & edge cases:
    - Uses vbt.RSI.run(close, window=14).rsi for RSI computation.
    - RSI NaNs (warmup) are filled with 50.0 to avoid early NaNs in signals.
    - Entry is generated when RSI crosses below 30 (prev >= 30 and curr < 30).
    - Exit is generated when RSI crosses above 70 (prev <= 70 and curr > 70).
    - Prevents double entries by tracking position state.
    - Ensures output has same length as input and contains no NaNs.
    """

    # Validate input structure
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]

    # Compute RSI using vectorbt (period 14)
    rsi_series = vbt.RSI.run(close, window=14).rsi

    # Fill initial NaNs from the RSI warmup with neutral 50.0 to avoid NaNs in logic
    rsi_filled = pd.Series.fillna(rsi_series, 50.0)

    # Previous RSI (shifted by 1). Use pd.Series.shift to comply with VAS requirements.
    rsi_prev = pd.Series.shift(rsi_filled, 1)

    # Detect crosses: cross below 30 (entry), cross above 70 (exit)
    cross_below = (rsi_prev >= 30.0) & (rsi_filled < 30.0)
    cross_above = (rsi_prev <= 70.0) & (rsi_filled > 70.0)

    # Build position series (0 or 1) without lookahead and avoiding double entries
    n = len(close)
    positions_array = np.zeros(n, dtype=np.int8)

    in_position = False
    # Use index-aligned iteration to preserve exact timestamps and avoid lookahead
    for i in range(n):
        # Boolean checks via .iloc to avoid any accidental use of future data
        is_entry = bool(cross_below.iloc[i])
        is_exit = bool(cross_above.iloc[i])

        if is_entry and not in_position:
            in_position = True
        elif is_exit and in_position:
            in_position = False

        positions_array[i] = 1 if in_position else 0

    positions = pd.Series(positions_array, index=close.index)

    # Ensure no NaNs in output (shouldn't be any) and correct length
    positions = pd.Series.fillna(positions, 0)

    return {"ohlcv": positions}
