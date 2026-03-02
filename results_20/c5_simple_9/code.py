# Auto-generated RSI mean reversion signal generator
# Implements generate_signals as specified in the prompt

from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """

    # Validate and extract data
    if data is None:
        raise ValueError("`data` must be a dict containing 'ohlcv' DataFrame")

    # Allow passing a DataFrame directly (some test harnesses may do this)
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("`data` dict must contain 'ohlcv' key with a DataFrame")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("`data` must be a pandas DataFrame or a dict with 'ohlcv' DataFrame")

    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("`ohlcv` must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("`ohlcv` DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].astype(float)

    # Extract and validate params with schema defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Helper: compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Use Wilder's smoothing: ewm with alpha=1/period is causal (no lookahead)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle cases where avg_loss == 0 -> RSI = 100, and avg_gain == 0 -> RSI = 0
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(~(avg_gain == 0), rsi)

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Define entry/exit signals based on RSI crosses
    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    # Exit:  RSI crosses above overbought (prev <= overbought and curr > overbought)

    prev_rsi = rsi.shift(1)

    entries = (prev_rsi >= oversold) & (rsi < oversold)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean type and alignment
    entries = entries.reindex(close.index, fill_value=False)
    exits = exits.reindex(close.index, fill_value=False)

    # Build position series (0 = flat, 1 = long) using state machine to avoid double entries
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_position = False

    # Iterate through index to maintain state (this is deterministic and causal)
    for i in range(len(close)):
        if entries.iloc[i] and not in_position:
            in_position = True
        elif exits.iloc[i] and in_position:
            in_position = False
        pos.iloc[i] = 1 if in_position else 0

    # Sanity: position must be 0 or 1
    pos = pos.astype(int)

    return {"ohlcv": pos}
