"""
RSI Mean Reversion Strategy - Signal Generator

Implements generate_signals(data, params) which returns a dict with key 'ohlcv'
mapping to a pandas Series of position targets (1 for long, 0 for flat).

Strategy logic:
- RSI period = 14 (default, can be overridden via params['rsi_window'])
- Go long when RSI crosses below 30 (prev >= 30 and current < 30)
- Exit when RSI crosses above 70 (prev <= 70 and current > 70)
- Long-only

The code uses vectorbt's RSI indicator: vbt.RSI.run(close, window=14)
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing market data. Expected to have key 'ohlcv'
            mapping to a DataFrame with a 'close' column.
        params: Strategy parameters. Recognized keys:
            - 'rsi_window' (int): RSI period. Defaults to 14.

    Returns:
        A dictionary with a single key 'ohlcv' mapping to a pandas Series of
        position targets (1 for long, 0 for flat), indexed the same as the
        input close series.

    Raises:
        ValueError: If required data is missing or invalid.
    """
    # Validate input
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].copy()
    if not isinstance(close, (pd.Series, pd.DataFrame)):
        raise ValueError("close must be a pandas Series or single-column DataFrame")

    # If close is a single-column DataFrame, convert to Series
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError('Multi-column close DataFrame is not supported')
        close = close.iloc[:, 0]

    if close.isna().all():
        raise ValueError('close series contains only NaN values')

    # Parameters
    rsi_window = int(params.get('rsi_window', 14)) if params is not None else 14
    if rsi_window <= 0:
        raise ValueError('rsi_window must be a positive integer')

    # Compute RSI using vectorbt
    rsi_ind = vbt.RSI.run(close, window=rsi_window)
    rsi = rsi_ind.rsi

    # rsi may be a DataFrame (multi-column). Ensure it's a Series for single asset.
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            raise ValueError('RSI returned multiple columns - multi-asset not supported')

    # Compute crossing signals
    prev_rsi = rsi.shift(1)

    # Only consider crossings when both previous and current RSI are finite
    valid = rsi.notna() & prev_rsi.notna()

    entry = (prev_rsi >= 30) & (rsi < 30) & valid
    exit = (prev_rsi <= 70) & (rsi > 70) & valid

    # Convert to boolean numpy arrays, treating NaNs as False
    entry_arr = entry.fillna(False).to_numpy(dtype=bool)
    exit_arr = exit.fillna(False).to_numpy(dtype=bool)

    # Build position array by iterating once (vectorized state machine)
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)
    in_long = False
    for i in range(n):
        if not in_long and entry_arr[i]:
            in_long = True
        elif in_long and exit_arr[i]:
            in_long = False
        pos_arr[i] = 1 if in_long else 0

    positions = pd.Series(pos_arr, index=close.index, name='position')

    return {'ohlcv': positions}
