from typing import Any, Dict

import pandas as pd
import numpy as np
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - RSI period = 14 (default)
    - Go long when RSI crosses below 30
    - Exit long when RSI crosses above 70
    - Long-only, single asset

    Args:
        data: Dictionary containing market data. Expected to contain key 'ohlcv'
              which should be a pandas DataFrame with a 'close' column (or a
              Series representing close prices).
        params: Dictionary of parameters. Recognized keys:
            - 'rsi_period' (int): RSI window (default 14)
            - 'entry_level' (float): RSI level to enter (default 30)
            - 'exit_level' (float): RSI level to exit (default 70)

    Returns:
        A dict with key 'ohlcv' mapped to a pandas Series of positions
        where 1 = long, 0 = flat. The Series index matches the input close index.

    Raises:
        ValueError: if input data is malformed.
    """

    # Basic validation
    if 'ohlcv' not in data:
        raise ValueError("Input data must contain 'ohlcv' key with close prices.")

    ohlcv = data['ohlcv']

    # Extract close series robustly
    if isinstance(ohlcv, pd.DataFrame):
        if 'close' in ohlcv.columns:
            close = ohlcv['close']
        else:
            # Try case-insensitive match for 'close'
            close_cols = [col for col in ohlcv.columns if str(col).lower() == 'close']
            if close_cols:
                close = ohlcv[close_cols[0]]
            elif ohlcv.shape[1] == 1:
                # Single-column DataFrame, assume it's the close
                close = ohlcv.iloc[:, 0]
            else:
                raise ValueError("'ohlcv' DataFrame must contain a 'close' column for single-asset backtest")
    elif isinstance(ohlcv, pd.Series):
        close = ohlcv
    else:
        raise ValueError("data['ohlcv'] must be a pandas DataFrame or Series")

    # Ensure we have a float Series and a proper index
    close = close.dropna().astype(float).copy()
    if close.empty:
        raise ValueError("Close price series is empty after dropping NaNs")

    # Parameters (use defaults if not provided)
    rsi_period = int(params.get('rsi_period', 14))
    entry_level = float(params.get('entry_level', 30))
    exit_level = float(params.get('exit_level', 70))

    # Compute RSI using vectorbt's RSI indicator
    # Verified signature: vbt.RSI.run(close, window=14) -> indicator with .rsi output
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    rsi = getattr(rsi_res, 'rsi', None)
    if rsi is None:
        raise RuntimeError("vbt.RSI.run did not produce 'rsi' output")

    # If RSI comes back as a DataFrame (e.g., multi-col input), squeeze to Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            # Multi-column RSI: pick the first column (single-asset expectation)
            rsi = rsi.iloc[:, 0]

    # Align RSI with close index (in case of earlier dropna)
    rsi = rsi.reindex(close.index)

    # Build crossing signals: entry when RSI crosses below entry_level; exit when crosses above exit_level
    rsi_prev = rsi.shift(1)
    entries = ((rsi_prev >= entry_level) & (rsi < entry_level)).fillna(False).astype(bool)
    exits = ((rsi_prev <= exit_level) & (rsi > exit_level)).fillna(False).astype(bool)

    # Convert boolean masks to a position series (0/1) by iterating through bars
    positions = pd.Series(0, index=close.index, dtype='int8')
    pos = 0
    for timestamp in close.index:
        if pos == 0 and entries.loc[timestamp]:
            pos = 1
        elif pos == 1 and exits.loc[timestamp]:
            pos = 0
        positions.loc[timestamp] = pos

    return {'ohlcv': positions}
