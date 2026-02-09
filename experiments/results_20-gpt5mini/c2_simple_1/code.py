import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy logic:
    - RSI period = 14 (can be overridden by params['rsi_period'])
    - Go long when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and curr > 70)

    Args:
        data: Dictionary containing at least the key 'ohlcv' mapped to a DataFrame
            with a 'close' column (pd.Series).
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI period (default 14)

    Returns:
        A dict with key 'ohlcv' containing a pd.Series of position targets:
        +1 for long, 0 for flat. Index matches input close series.

    Raises:
        ValueError: If required data is missing or malformed.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close']
    if not isinstance(close, (pd.Series, pd.DataFrame)):
        raise ValueError("ohlcv['close'] must be a pandas Series or DataFrame column")

    # Use single-series close (if DataFrame column returned as DataFrame, extract first col)
    if isinstance(close, pd.DataFrame):
        # If it's a single column DataFrame, convert to Series
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("Expected a single 'close' series, got a DataFrame with multiple columns")

    # Parameters
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    # Compute RSI using vectorbt's RSI indicator
    # vbt.RSI.run returns an object with attribute `.rsi`
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # Align types
    rsi = rsi.reindex(close.index)

    # Prepare signals: detect crossings
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below 30 (prev >= 30 and curr < 30)
    entry_signals = (prev_rsi >= 30) & (rsi < 30)
    # Exit: RSI crosses above 70 (prev <= 70 and curr > 70)
    exit_signals = (prev_rsi <= 70) & (rsi > 70)

    # Ensure boolean Series (fill NaN with False)
    entry_signals = entry_signals.fillna(False).astype(bool)
    exit_signals = exit_signals.fillna(False).astype(bool)

    # Build position series: +1 for long, 0 for flat
    positions = pd.Series(0, index=close.index, dtype='int8')

    in_position = False
    # Iterate through timestamps to apply signals sequentially
    for idx in close.index:
        if not in_position and entry_signals.loc[idx]:
            in_position = True
        elif in_position and exit_signals.loc[idx]:
            in_position = False
        positions.loc[idx] = 1 if in_position else 0

    return {"ohlcv": positions}
