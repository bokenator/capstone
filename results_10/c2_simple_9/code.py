from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate long-only position targets based on RSI mean reversion.

    Strategy details:
    - RSI period = 14 (configurable via params['rsi_period'])
    - Go long when RSI crosses below oversold level (default 30)
    - Exit when RSI crosses above overbought level (default 70)

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv' -> pd.DataFrame
              with at least a 'close' column.
        params: Dictionary of parameters. Supported keys:
            - rsi_period: int, lookback for RSI (default 14)
            - oversold: float, oversold threshold (default 30.0)
            - overbought: float, overbought threshold (default 70.0)

    Returns:
        A dict with key 'ohlcv' mapping to a pd.Series of position targets
        where 1 = long, 0 = flat. The series index matches the input close prices.
    """
    # Validate inputs
    if "ohlcv" not in data:
        raise ValueError("`data` must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("`data['ohlcv']` must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("`ohlcv` DataFrame must contain a 'close' column")

    close = ohlcv["close"]
    # If close is a single-column DataFrame, convert to Series
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("`ohlcv['close']` must be a Series or single-column DataFrame")

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using vectorbt indicator
    # vbt.RSI.run returns an object with attribute `.rsi` (Series or DataFrame)
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi
    # If RSI comes back as DataFrame (multi-column), take the first column
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] >= 1:
            rsi = rsi.iloc[:, 0]
        else:
            raise ValueError("RSI indicator returned empty DataFrame")

    # Build boolean entry/exit signals using cross logic
    prev_rsi = rsi.shift(1)

    # Entry: previous RSI was >= oversold and current RSI is < oversold (crossed below)
    entry = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit: previous RSI was <= overbought and current RSI is > overbought (crossed above)
    exit = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN with False for safety
    entry = entry.reindex(rsi.index).fillna(False)
    exit = exit.reindex(rsi.index).fillna(False)

    # Construct a forward-filled action series: set 1.0 on entry, 0.0 on exit, then ffill
    action = pd.Series(data=np.nan, index=rsi.index, dtype=float)
    action[entry] = 1.0
    action[exit] = 0.0

    # Forward fill to maintain position until an exit occurs, start flat (0)
    positions = action.ffill().fillna(0.0).astype(int)

    # Ensure positions index matches the original close index
    positions = positions.reindex(close.index).fillna(0).astype(int)

    return {"ohlcv": positions}
