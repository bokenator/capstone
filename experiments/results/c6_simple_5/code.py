import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean-reversion strategy.

    Strategy logic:
    - RSI period = 14
    - Go long when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit long when RSI crosses above 70 (prev <= 70 and curr > 70)
    - Long-only. Position values: 1 for long, 0 for flat.

    Args:
        data: Dictionary containing "ohlcv" DataFrame with a "close" column.
        params: Parameters dict (not used for now, kept for API compatibility).

    Returns:
        Dict with key "ohlcv" mapping to a pd.Series of position targets.
    """
    # Extract close prices
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]

    # Calculate RSI using vectorbt
    # Use the verified API: fully-qualified call
    rsi_raw = vbt.RSI.run(close, window=14).rsi

    # Fill initial NaNs to avoid propagation issues. Use forward/backward fill
    # via module-qualified pandas calls from the VAS.
    rsi = pd.Series.ffill(rsi_raw)
    rsi = pd.Series.bfill(rsi)
    rsi = pd.Series.fillna(rsi, 50.0)  # fallback to neutral 50 if still NaN

    # Previous RSI (lag 1) using module-qualified shift
    prev = pd.Series.shift(rsi, 1)

    # Vectorized values for fast iteration
    rsi_vals = rsi.values
    prev_vals = prev.values

    n = len(rsi_vals)
    positions = np.zeros(n, dtype=np.int8)

    # Simulate position to avoid double entries/exits
    for i in range(n):
        # Current and previous RSI; handle NaN in prev
        curr_rsi = rsi_vals[i]
        prev_rsi = prev_vals[i]

        is_prev_valid = not np.isnan(prev_rsi)
        entry_cond = False
        exit_cond = False

        if is_prev_valid:
            entry_cond = (prev_rsi >= 30.0) and (curr_rsi < 30.0)
            exit_cond = (prev_rsi <= 70.0) and (curr_rsi > 70.0)

        if i == 0:
            positions[i] = 0
            continue

        # Carry previous position by default
        positions[i] = positions[i - 1]

        if positions[i - 1] == 0 and entry_cond:
            # Enter long
            positions[i] = 1
        elif positions[i - 1] == 1 and exit_cond:
            # Exit to flat
            positions[i] = 0

    # Convert to pandas Series with same index as input
    position_series = pd.Series(positions, index=rsi.index)

    return {"ohlcv": position_series}
