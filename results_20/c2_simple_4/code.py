from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position series using an RSI mean reversion strategy.

    Strategy rules:
    - RSI period = 14 (configurable via params['rsi_period'])
    - Go long when RSI crosses below the oversold threshold (default 30)
    - Exit (go flat) when RSI crosses above the overbought threshold (default 70)

    Args:
        data: Dictionary containing market data. Must contain key 'ohlcv' with a
            DataFrame that has a 'close' column (pd.Series).
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'oversold' (float): Oversold threshold. Default 30.
            - 'overbought' (float): Overbought threshold. Default 70.

    Returns:
        A dictionary with key 'ohlcv' mapping to a pd.Series of target positions
        aligned with the input close series. Values are 1 for long and 0 for flat.

    Notes:
        - Handles NaNs / warmup periods by keeping positions flat until RSI is
          valid and a crossing occurs.
        - This function is intentionally simple and deterministic to work with
          the provided backtest runner which converts position changes into
          entries/exits.
    """

    # Validate input structure
    if "ohlcv" not in data:
        raise ValueError("Input data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"]

    # If close is a DataFrame (e.g., multiple assets), reduce to a single series
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            # If multiple columns are present, pick the first column.
            # This strategy is intended for a single asset.
            close = close.iloc[:, 0]

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using vectorbt's indicator
    # vbt.RSI.run returns an object with `.rsi` attribute (see vectorbt docs)
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # If rsi is a DataFrame (rare for single-series input), reduce to a Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            rsi = rsi.iloc[:, 0]

    # Align RSI with close index
    rsi = rsi.reindex(close.index)

    # Define crossings: entry when RSI crosses below oversold; exit when RSI crosses above overbought
    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    # Guard against NaNs in the warmup period
    entry_cond = entry_cond.fillna(False)
    exit_cond = exit_cond.fillna(False)

    # Build position by counting entries and exits and ensuring non-negative position count
    # Position is 1 when the count of entries exceeds the count of exits, else 0
    position_count = (entry_cond.cumsum() - exit_cond.cumsum()).clip(lower=0)
    position = (position_count > 0).astype(int)

    # Ensure the returned series is aligned and contains no NaNs
    position = position.reindex(close.index).fillna(0).astype(int)

    return {"ohlcv": position}
