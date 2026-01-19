from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(
    data: Union[Dict[str, pd.DataFrame], pd.Series],
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, pd.Series], Tuple[pd.Series, pd.Series]]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Behavior:
    - Calculate RSI with period = 14 using vectorbt
    - Entry when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and curr > 70)
    - Ensure no double-entry: entries only when not already long

    Accepts either:
    - data: dict with key "ohlcv" containing a DataFrame with a 'close' column
      -> returns {'ohlcv': position_series} where position_series values are 0 or 1
    - data: pd.Series of close prices
      -> returns (entries_series, exits_series) boolean Series representing actual trade signals

    Args:
        data: price data (dict or Series)
        params: unused (kept for compatibility)

    Returns:
        Dict or tuple as described above
    """

    # Extract close series from input (support both interfaces)
    if isinstance(data, pd.Series):
        close = data
        input_is_series = True
    elif isinstance(data, dict) and "ohlcv" in data and "close" in data["ohlcv"]:
        close = data["ohlcv"]["close"]
        input_is_series = False
    else:
        raise ValueError("data must be a pd.Series or a dict with key 'ohlcv' containing a 'close' column")

    # Compute RSI using vectorbt (period=14)
    rsi_series = vbt.RSI.run(close, window=14).rsi

    # Previous RSI (shift by 1) - use function-style call to avoid instance method qualification issues
    prev_rsi = pd.Series.shift(rsi_series, 1)

    # Convert to numpy arrays for robust elementwise logic (NaNs in prev_rsi will compare False)
    prev_vals = prev_rsi.values
    curr_vals = rsi_series.values

    # Entry: prev >= 30 and curr < 30 (cross below 30)
    entry_signals_raw = (prev_vals >= 30) & (curr_vals < 30)

    # Exit: prev <= 70 and curr > 70 (cross above 70)
    exit_signals_raw = (prev_vals <= 70) & (curr_vals > 70)

    # Build position series (0 or 1), ensuring no double-entry
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)
    in_position = False
    current = 0
    for i in range(n):
        # If raw entry signal and not already in position -> enter
        if entry_signals_raw[i] and (not in_position):
            current = 1
            in_position = True
        # If raw exit signal and currently in position -> exit
        elif exit_signals_raw[i] and in_position:
            current = 0
            in_position = False
        # else current remains
        pos_arr[i] = current

    # Construct position series aligned with input index
    position_series = pd.Series(np.array(pos_arr, dtype=np.int8), index=close.index)

    # Derive entry/exit signals from position changes (for the Series-returning interface)
    pos_diff = pd.Series.diff(position_series)
    pos_diff_filled = pd.Series.fillna(pos_diff, 0)

    entries_series = pos_diff_filled > 0
    exits_series = pos_diff_filled < 0

    if input_is_series:
        # Return entries/exits as boolean Series
        return entries_series, exits_series

    # For dict input (backtester), return the position series under 'ohlcv'
    return {"ohlcv": position_series}
