from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy:
    - Compute RSI with period = 14 (default, can be overridden via params['rsi_period'])
    - Go long when RSI crosses below the oversold level (default 30)
    - Exit when RSI crosses above the overbought level (default 70)

    Args:
        data: Dict containing market data. Expected to contain key 'ohlcv' -> DataFrame
              with a 'close' column (pd.Series).
        params: Dict of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period (default 14)
            - 'rsi_entry' (float): Oversold threshold for entry (default 30)
            - 'rsi_exit'  (float): Overbought threshold for exit (default 70)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (1 = long, 0 = flat).

    Notes:
        - Warm-up periods (NaNs from RSI) are handled: no entries/exits are generated while RSI
          or its lag is NaN.
        - Only single-asset (one close series) is supported.
    """
    # Basic validation
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping names to DataFrames (expected 'ohlcv').")
    if 'ohlcv' not in data:
        raise KeyError("`data` must contain an 'ohlcv' DataFrame with a 'close' column.")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame containing OHLCV columns.")
    if 'close' not in ohlcv.columns:
        raise KeyError("`data['ohlcv']` must contain a 'close' column.")

    close = ohlcv['close']

    # If close is a DataFrame with a single column, convert to Series
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("Only single-asset strategies are supported. Provide a single 'close' series.")

    # Parameters with sensible defaults
    rsi_period = int(params.get('rsi_period', 14))
    entry_level = float(params.get('rsi_entry', 30.0))
    exit_level = float(params.get('rsi_exit', 70.0))

    if not (0 < entry_level < 100) or not (0 < exit_level < 100):
        raise ValueError("RSI thresholds must be between 0 and 100.")

    # Compute RSI using vectorbt's indicator
    # vbt.RSI.run returns an object with attribute `rsi` (a Series aligned with `close`)
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # Avoid generating signals on NaN values (warmup)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below entry_level (from >= entry_level to < entry_level)
    entries = prev_rsi.notna() & rsi.notna() & (prev_rsi >= entry_level) & (rsi < entry_level)

    # Exit: RSI crosses above exit_level (from <= exit_level to > exit_level)
    exits = prev_rsi.notna() & rsi.notna() & (prev_rsi <= exit_level) & (rsi > exit_level)

    # Build a position series using a simple state machine implemented via forward-fill
    actions = pd.Series(data=np.nan, index=close.index, dtype=float)
    actions[entries] = 1.0
    actions[exits] = 0.0

    # Forward-fill actions to hold positions between entry and exit; default to flat (0)
    positions = actions.ffill().fillna(0.0).astype(int)

    return {"ohlcv": positions}
