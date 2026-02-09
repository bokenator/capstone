from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - Calculate RSI with period = params['rsi_period']
    - Go long when RSI crosses below params['oversold']
    - Exit when RSI crosses above params['overbought']
    - Long-only: positions are 1 (long) or 0 (flat)

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Example: {"ohlcv": pd.Series([...], index=...)}
    """

    # Basic validations
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters (use provided defaults if missing)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Parameter sanity checks (within declared PARAM_SCHEMA ranges)
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI using Wilder's smoothing (recursive EMA with adjust=False)
    # This implementation uses only past data up to time t to compute RSI at t (no lookahead).
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / float(rsi_period)
    # adjust=False ensures recursive calculation (no lookahead)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss == 0 (RSI = 100) or avg_gain == 0 (RSI = 0)
    rsi = rsi.copy()
    rsi[avg_loss == 0.0] = 100.0
    rsi[avg_gain == 0.0] = 0.0

    # Ensure index alignment
    rsi.index = close.index

    # Entry: RSI crosses below oversold (previous >= oversold, current < oversold)
    prev_rsi = rsi.shift(1)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (previous <= overbought, current > overbought)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Build long-only position series by iterating through time (state machine)
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = 0
    # Use integer location for speed and to avoid ambiguity with indexes
    for i in range(len(position)):
        if in_position == 0 and entry_signals.iat[i]:
            in_position = 1
        elif in_position == 1 and exit_signals.iat[i]:
            in_position = 0
        position.iat[i] = in_position

    return {"ohlcv": position}
