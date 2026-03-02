"""
RSI Mean Reversion Signal Generator

This module exports a single function `generate_signals` which computes RSI-based
mean reversion signals for a single-asset, long-only strategy.

Strategy logic:
- RSI period = 14 (default, configurable via params)
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)

The function returns a dict with key 'ohlcv' that maps to a pandas Series of
position targets: 1 = long, 0 = flat. The Series index matches the provided
close price series.

The implementation is robust to NaNs and warmup periods.
"""

from typing import Dict

import numpy as np
import pandas as pd


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate long-only position targets based on RSI mean reversion.

    Args:
        data: Dictionary containing market data. Expected to contain key
            'ohlcv' mapped to a pandas DataFrame with a 'close' column.
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'rsi_lower' (float): RSI oversold threshold. Default 30.
            - 'rsi_upper' (float): RSI overbought threshold. Default 70.

    Returns:
        Dict with a single key 'ohlcv' mapping to a pd.Series of integer
        position targets (1 for long, 0 for flat). The Series index matches
        the input close price index.

    Raises:
        TypeError / ValueError for invalid inputs.
    """
    # Input validation
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing 'ohlcv' DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame value")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close']

    # If close is a DataFrame (multi-column), pick the first column
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            raise ValueError("close DataFrame contains no columns")
        close = close.iloc[:, 0]

    if not isinstance(close, pd.Series):
        # coerce to series
        close = pd.Series(close)

    # Handle trivial case
    if len(close) == 0:
        # return empty series
        return {'ohlcv': pd.Series(dtype='int8')}

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    rsi_lower = float(params.get('rsi_lower', 30.0)) if params is not None else 30.0
    rsi_upper = float(params.get('rsi_upper', 70.0)) if params is not None else 70.0

    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    # Robust to Series input and handles NaNs gracefully.
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use ewm for Wilder smoothing
    ma_up = up.ewm(alpha=1.0 / float(rsi_period), adjust=False).mean()
    ma_down = down.ewm(alpha=1.0 / float(rsi_period), adjust=False).mean()

    # Compute RS and RSI. Use replace(0, np.nan) to avoid division-by-zero warnings.
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If both ma_up and ma_down are zero (flat price), set RSI to neutral 50.
    flat_mask = (ma_up == 0.0) & (ma_down == 0.0)
    if flat_mask.any():
        rsi = rsi.mask(flat_mask, 50.0)

    # At this point, warmup periods will still be NaN in rsi. We'll keep them as NaN
    # and ensure positions remain flat during NaNs.

    # Entry: RSI crosses below rsi_lower (previous >= lower and current < lower)
    prev_rsi = rsi.shift(1)
    entry = (prev_rsi >= rsi_lower) & (rsi < rsi_lower)

    # Exit: RSI crosses above rsi_upper (previous <= upper and current > upper)
    exit = (prev_rsi <= rsi_upper) & (rsi > rsi_upper)

    # Build position series (stateful): 1 when in long, 0 when flat.
    pos_values = np.zeros(len(rsi), dtype='int8')
    in_position = False

    # Use integer positional indexing for speed and to avoid index alignment issues
    entry_vals = entry.fillna(False).values
    exit_vals = exit.fillna(False).values

    for i in range(len(rsi)):
        if not in_position and entry_vals[i]:
            in_position = True
        elif in_position and exit_vals[i]:
            in_position = False
        pos_values[i] = 1 if in_position else 0

    positions = pd.Series(pos_values, index=rsi.index, name='position')

    # Ensure no positions during RSI NaN (warmup), keep flat if RSI is NaN
    positions[rsi.isna()] = 0

    # Return mapping expected by the backtest runner
    return {'ohlcv': positions}
