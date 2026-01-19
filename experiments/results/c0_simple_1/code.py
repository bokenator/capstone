from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy:
    - RSI period = 14 (default; can be overridden via params['rsi_period'])
    - Enter long when RSI crosses below oversold level (default 30)
    - Exit long when RSI crosses above overbought level (default 70)
    - Long-only, single asset

    Args:
        data: Dict containing at least the key 'ohlcv' with a DataFrame that has a
              'close' column (pd.Series).
        params: Dict of parameters. Supported keys:
            - rsi_period: int = 14
            - oversold: float = 30.0
            - overbought: float = 70.0

    Returns:
        A dict with key 'ohlcv' mapping to a pd.Series or pd.DataFrame of target
        positions (1 for long, 0 for flat), aligned with the input close prices.
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"]

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using vectorbt's indicator (handles Series/DataFrame)
    # The .rsi attribute returns the RSI values aligned with close
    rsi_obj = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_obj.rsi

    # Ensure alignment and handle NaNs: shift for crossing detection
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below the oversold threshold (prev >= oversold and current < oversold)
    entries = (rsi_prev >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above the overbought threshold (prev <= overbought and current > overbought)
    exits = (rsi_prev <= overbought) & (rsi > overbought)

    # Replace NaN booleans with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Convert booleans to integers and compute running position count
    # Works for both Series and DataFrame inputs
    cum_entries = entries.astype(int).cumsum()
    cum_exits = exits.astype(int).cumsum()

    raw_pos = (cum_entries - cum_exits).clip(lower=0, upper=1)

    # Ensure position type is numeric (float) and aligned
    position = raw_pos.astype(float)

    return {"ohlcv": position}
