from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate trading signals (position targets) for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute RSI with period `rsi_window` (default 14)
    - Go long (position = 1) when RSI crosses below `rsi_lower` (default 30)
    - Exit long (position = 0) when RSI crosses above `rsi_upper` (default 70)
    - Long-only, single asset

    Args:
        data: A dictionary containing an 'ohlcv' pandas DataFrame with a 'close' column.
        params: Strategy parameters. Supported keys:
            - 'rsi_window' (int): RSI lookback window. Default 14.
            - 'rsi_lower' (float): Oversold threshold. Default 30.
            - 'rsi_upper' (float): Overbought threshold. Default 70.

    Returns:
        A dict with key 'ohlcv' mapping to a pandas Series of position targets (1 for long, 0 for flat).
    """

    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict containing an 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("`data` must contain key 'ohlcv' with a pandas DataFrame")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("`data['ohlcv']` must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("`data['ohlcv']` must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with defaults
    rsi_window = int(params.get('rsi_window', 14)) if params is not None else 14
    rsi_lower = float(params.get('rsi_lower', 30.0)) if params is not None else 30.0
    rsi_upper = float(params.get('rsi_upper', 70.0)) if params is not None else 70.0

    if rsi_window < 1:
        raise ValueError("rsi_window must be >= 1")
    if not (0 <= rsi_lower < rsi_upper <= 100):
        raise ValueError("rsi_lower and rsi_upper must satisfy 0 <= lower < upper <= 100")

    # Compute RSI using vectorbt's indicator
    # Verified API: vbt.RSI.run(close, window=14) -> returns an indicator result with `.rsi` output
    rsi_ind = vbt.RSI.run(close, window=rsi_window)

    # Use helper methods to detect crossings
    # Verified APIs: rsi_ind.rsi_crossed_below(value), rsi_ind.rsi_crossed_above(value)
    entries = rsi_ind.rsi_crossed_below(rsi_lower)
    exits = rsi_ind.rsi_crossed_above(rsi_upper)

    # Align indices and coerce to boolean (no NaNs)
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = exits.reindex(close.index).fillna(False).astype(bool)

    # Build position series (1 = long, 0 = flat) using a simple state machine
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    # Iterate through bars to set position targets
    for i in range(len(close)):
        if not in_position:
            if entries.iloc[i]:
                in_position = True
                position.iloc[i] = 1
            else:
                position.iloc[i] = 0
        else:
            if exits.iloc[i]:
                in_position = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1

    return {"ohlcv": position}
