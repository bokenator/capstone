"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean-reversion strategy:
- RSI period: 14 (configurable via params['rsi_period'])
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)
- Long-only, single asset

Exports:
- generate_signals(data, params) -> dict[str, pd.Series]

The returned dict must contain the key 'ohlcv' mapped to a pd.Series of
position targets: 1 for long, 0 for flat. This is compatible with the
provided backtest runner which converts position diffs into entries/exits
for vectorbt.Portfolio.from_signals.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position targets based on RSI mean reversion.

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv'
            with a pandas.DataFrame that has a 'close' column.
        params: Dictionary of parameters. Recognized keys:
            - 'rsi_period' (int): RSI lookback period. Defaults to 14.

    Returns:
        A dictionary with a single key 'ohlcv' mapping to a pandas.Series of
        position targets (1 = long, 0 = flat). The Series index matches the
        input close price index.

    Raises:
        ValueError: If required data is missing or malformed.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing an 'ohlcv' DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain a 'close' column")

    close = ohlcv['close'].copy()

    # Parameters
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14

    # Compute RSI using vectorbt's built-in indicator
    # vbt.RSI.run returns an object with attribute `.rsi` (Series or DataFrame)
    rsi_out = vbt.RSI.run(close, window=rsi_period).rsi

    # If RSI returned a DataFrame (e.g., multi-column input), pick the first column
    if isinstance(rsi_out, pd.DataFrame):
        if rsi_out.shape[1] > 1:
            # Strategy is single-asset; select the first column
            rsi = rsi_out.iloc[:, 0].copy()
        else:
            rsi = rsi_out.iloc[:, 0].copy()
    else:
        rsi = rsi_out.copy()

    # Ensure RSI aligns with close index
    rsi = rsi.reindex(close.index)

    # Build crossing signals
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below 30 (prev >= 30 and current < 30)
    entries = (rsi_prev >= 30) & (rsi < 30)

    # Exit: RSI crosses above 70 (prev <= 70 and current > 70)
    exits = (rsi_prev <= 70) & (rsi > 70)

    # Ensure boolean series and align indices
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Simulate position targets (1 = long, 0 = flat)
    length = len(close)
    entry_vals = entries.to_numpy(dtype=bool)
    exit_vals = exits.to_numpy(dtype=bool)

    pos_vals = np.zeros(length, dtype=np.int8)
    state = 0  # 0 = flat, 1 = long
    for i in range(length):
        if state == 0 and entry_vals[i]:
            state = 1
        elif state == 1 and exit_vals[i]:
            state = 0
        pos_vals[i] = state

    positions = pd.Series(pos_vals, index=close.index, name='position')

    # Return signals dict compatible with the backtest runner
    return {'ohlcv': positions}
