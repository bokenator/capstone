"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean-reversion strategy:
- RSI period: 14 (default; can be overridden by params['rsi_period'])
- Go long when RSI crosses below 30
- Exit long when RSI crosses above 70
- Long-only, single-asset

Provides a single function `generate_signals` that the backtest runner will call.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets for an RSI mean-reversion strategy.

    Args:
        data: Dictionary containing market data. Expects key 'ohlcv' mapping to a
            pd.DataFrame with a 'close' column (single asset).
        params: Parameters dictionary. Supports optional key 'rsi_period' (int).

    Returns:
        A dictionary with key 'ohlcv' mapping to a pd.Series of position targets
        with the same index as the input close series. Values are 1 for long
        and 0 for flat.

    Raises:
        ValueError: If required data is missing or multiple assets are provided.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close']

    # If close is a DataFrame (e.g., shape (n,1)), ensure single asset
    if isinstance(close, pd.DataFrame):
        if close.shape[1] > 1:
            raise ValueError("This strategy supports a single asset only")
        # squeeze to Series
        close = close.iloc[:, 0]

    # Ensure series
    if not isinstance(close, pd.Series):
        # try to coerce
        close = pd.Series(close)

    # Parameters
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    rsi_oversold = float(params.get('rsi_oversold', 30.0)) if params is not None else 30.0
    rsi_overbought = float(params.get('rsi_overbought', 70.0)) if params is not None else 70.0

    # Compute RSI using vectorbt's RSI indicator
    # vbt.RSI.run returns an object with helpful methods like rsi_crossed_below/above
    rsi_ind = vbt.RSI.run(close, window=rsi_period)

    # Use indicator helpers to detect crosses
    entries = rsi_ind.rsi_crossed_below(rsi_oversold)
    exits = rsi_ind.rsi_crossed_above(rsi_overbought)

    # If entries/exits came back as DataFrame with single column, squeeze to Series
    if isinstance(entries, pd.DataFrame):
        if entries.shape[1] > 1:
            raise ValueError("This strategy supports a single asset only")
        entries = entries.iloc[:, 0]
    if isinstance(exits, pd.DataFrame):
        if exits.shape[1] > 1:
            raise ValueError("This strategy supports a single asset only")
        exits = exits.iloc[:, 0]

    # Align indices (safety)
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = exits.reindex(close.index).fillna(False).astype(bool)

    # Build position series from event markers:
    # - Put 1 at entry dates, 0 at exit dates
    # - Forward fill to maintain position until an exit
    markers = pd.Series(data=np.nan, index=close.index, dtype=float)
    markers.loc[entries] = 1.0
    markers.loc[exits] = 0.0

    # Forward-fill markers to produce positions; start flat (0) before first marker
    positions = markers.ffill().fillna(0).astype(int)

    # Ensure index name/type preserved and return dict as expected by backtester
    return {"ohlcv": positions}
