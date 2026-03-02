import pandas as pd
import numpy as np
import vectorbt as vbt

from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Strategy Logic
    - Calculate RSI with period = 14 (configurable via params)
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dictionary containing at least the key 'ohlcv' mapped to a DataFrame
              with a 'close' column (pd.Series).
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI window length. Default: 14
            - 'rsi_entry' (float): Entry threshold (oversold). Default: 30
            - 'rsi_exit' (float): Exit threshold (overbought). Default: 70

    Returns:
        A dictionary with key 'ohlcv' containing a pd.Series of positions aligned
        with the input close prices. Values: 1 for long, 0 for flat.
    """

    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise ValueError("data must contain key 'ohlcv' mapping to an OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close']
    if isinstance(close, pd.DataFrame):
        # Expect a single column Series for a single asset
        if close.shape[1] != 1:
            raise ValueError("Only single-asset data is supported: 'close' must be a Series or single-column DataFrame")
        close = close.iloc[:, 0]

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    rsi_entry = float(params.get('rsi_entry', 30.0))
    rsi_exit = float(params.get('rsi_exit', 70.0))

    # Calculate RSI using vectorbt's RSI indicator
    # vbt.RSI.run returns an indicator result with attribute `.rsi`
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_res.rsi

    # If RSI returned a DataFrame (multi-column), reduce to a Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] != 1:
            raise ValueError("RSI indicator returned multiple columns; only single-asset supported")
        rsi = rsi.iloc[:, 0]

    # Build crossing signals
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below entry threshold (previous >= entry, current < entry)
    entry_signal = (prev_rsi >= rsi_entry) & (rsi < rsi_entry)

    # Exit: RSI crosses above exit threshold (previous <= exit, current > exit)
    exit_signal = (prev_rsi <= rsi_exit) & (rsi > rsi_exit)

    # Replace NA with False for signal logic
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Simulate position state machine (long-only)
    position = pd.Series(0, index=close.index, dtype=int)
    in_long = False

    # Use integer positions: 1 for long, 0 for flat
    for i in range(len(close.index)):
        if not in_long and bool(entry_signal.iloc[i]):
            in_long = True
            position.iloc[i] = 1
        elif in_long and bool(exit_signal.iloc[i]):
            in_long = False
            position.iloc[i] = 0
        else:
            position.iloc[i] = 1 if in_long else 0

    # Ensure position is aligned with close index and has no NaNs
    position = position.reindex(close.index).fillna(0).astype(int)

    return {"ohlcv": position}
