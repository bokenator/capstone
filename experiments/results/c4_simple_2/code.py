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
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series. Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate inputs
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close']
    if not isinstance(close, pd.Series):
        # Ensure close is a pandas Series
        close = pd.Series(close)

    # Validate and extract params (only allowed keys)
    if 'rsi_period' in params:
        rsi_period = int(params['rsi_period'])
    else:
        rsi_period = 14

    if 'oversold' in params:
        oversold = float(params['oversold'])
    else:
        oversold = 30.0

    if 'overbought' in params:
        overbought = float(params['overbought'])
    else:
        overbought = 70.0

    # Basic bounds checking
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    if not (0.0 <= oversold <= 50.0):
        raise ValueError('oversold must be between 0 and 50')
    if not (50.0 <= overbought <= 100.0):
        raise ValueError('overbought must be between 50 and 100')

    # Calculate RSI using vectorbt
    rsi_result = vbt.RSI.run(close, window=rsi_period).rsi

    # Ensure we have a Series (single asset)
    if isinstance(rsi_result, pd.DataFrame):
        # Take the first column if a DataFrame is returned
        rsi = rsi_result.iloc[:, 0]
    else:
        rsi = rsi_result

    # Prepare previous RSI (shifted by 1)
    prev_rsi = pd.Series.shift(rsi, 1)

    # Fill NA for comparisons: use infinities so comparisons behave consistently
    prev_rsi_f = pd.Series.fillna(prev_rsi, np.inf)
    rsi_f = pd.Series.fillna(rsi, np.inf)

    # Entry: RSI crosses below oversold (prev > oversold and current <= oversold)
    entry_mask = (prev_rsi_f > oversold) & (rsi_f <= oversold)

    # Exit: RSI crosses above overbought (prev < overbought and current >= overbought)
    exit_mask = (prev_rsi_f < overbought) & (rsi_f >= overbought)

    # Construct position series (long-only). Iterate sequentially to maintain state.
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)
    entry_vals = entry_mask.values
    exit_vals = exit_mask.values

    state = 0  # 0 = flat, 1 = long
    for i in range(n):
        # Skip if rsi is NaN at this bar (no reliable signal)
        if not np.isfinite(rsi.values[i]):
            # keep previous state
            pos_arr[i] = state
            continue

        if state == 0:
            if bool(entry_vals[i]):
                state = 1
        else:
            if bool(exit_vals[i]):
                state = 0

        pos_arr[i] = state

    position_series = pd.Series(pos_arr, index=close.index, dtype=np.int8)

    return {"ohlcv": position_series}
