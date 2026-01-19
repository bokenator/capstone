from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy rules:
    - RSI period = 14
    - Enter long when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit long when RSI crosses above 70 (prev <= 70 and curr > 70)
    - Long-only, single asset

    Args:
        data: Dictionary containing an 'ohlcv' DataFrame with a 'close' column.
        params: Parameters dict (unused for now but kept for interface compatibility).

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of position targets (1 for long, 0 for flat).
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]

    # Calculate RSI using vectorbt (period fixed to 14)
    rsi_obj = vbt.RSI.run(close, window=14)
    rsi = rsi_obj.rsi

    # Prepare numpy arrays for signal computation
    # Use raw values and check finite values to avoid NaN-driven false signals
    curr_vals = rsi.values
    prev_vals = rsi.shift(1).values

    # Conditions: cross below 30 for entry, cross above 70 for exit
    is_finite_prev = np.isfinite(prev_vals)
    is_finite_curr = np.isfinite(curr_vals)

    entry_cond = is_finite_prev & is_finite_curr & (prev_vals >= 30.0) & (curr_vals < 30.0)
    exit_cond = is_finite_prev & is_finite_curr & (prev_vals <= 70.0) & (curr_vals > 70.0)

    # Build positions by walking through time (stateful, long-only)
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)
    long_state = 0

    # Convert conditions to plain boolean numpy arrays
    entry_bool = np.where(entry_cond, True, False)
    exit_bool = np.where(exit_cond, True, False)

    for i in range(n):
        if long_state == 0:
            # Not in a position: enter if entry signal
            if entry_bool[i]:
                long_state = 1
        else:
            # In a position: exit if exit signal
            if exit_bool[i]:
                long_state = 0
        pos_arr[i] = long_state

    # Convert to pandas Series aligned with the input index
    position = pd.Series(pos_arr, index=close.index)

    return {"ohlcv": position}
