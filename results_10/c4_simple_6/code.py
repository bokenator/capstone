from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - Calculate RSI with period = params['rsi_period']
    - Go long when RSI CROSSES BELOW params['oversold']
    - Exit when RSI CROSSES ABOVE params['overbought']
    - Long-only, single asset (returns positions 0 or 1)

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with a DataFrame that has a 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Example: {"ohlcv": pd.Series([...], index=...)}

    Notes:
        - This implementation uses Wilder's smoothing (EMA with alpha=1/period) for RSI.
        - Early periods where RSI cannot be computed will not generate signals and the
          strategy remains flat (position = 0).
    """
    # Validate input data
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Validate params and apply defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Prevent division by zero; where roll_down == 0 and roll_up > 0, RSI -> 100
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Compute crossing signals
    rsi_prev = rsi.shift(1)

    entry_signals = (rsi_prev >= oversold) & (rsi < oversold)
    exit_signals = (rsi_prev <= overbought) & (rsi > overbought)

    # Treat NaNs as no-signal
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series (0 = flat, 1 = long)
    pos_arr = np.zeros(len(close), dtype=np.int8)
    in_position = False

    # Use integer indexing to iterate in order - preserves DatetimeIndex mapping
    for i in range(len(close)):
        if not in_position:
            if bool(entry_signals.iloc[i]):
                in_position = True
                pos_arr[i] = 1
            else:
                pos_arr[i] = 0
        else:
            if bool(exit_signals.iloc[i]):
                in_position = False
                pos_arr[i] = 0
            else:
                pos_arr[i] = 1

    position = pd.Series(pos_arr, index=close.index, name='position')

    return {'ohlcv': position}
