import pandas as pd
import numpy as np
from typing import Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
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

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Extract and validate params (use only allowed params)
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Helper: compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    # Handles NaNs in the input close series gracefully.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use ewm with adjust=False for Wilder's smoothing and min_periods for warmup
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Compute RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 -> RSI = 100 (no losses)
    # - If avg_gain == 0 and avg_loss == 0 -> RSI = 50 (no price movement)
    rsi = rsi.astype(float)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi = rsi.mask(avg_loss == 0, 100.0)

    # Determine entry (crosses below oversold) and exit (crosses above overbought)
    prev_rsi = rsi.shift(1)

    entries = (prev_rsi >= oversold) & (rsi < oversold)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series and no NaNs
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Build position series: long (1) when cumulative entries > cumulative exits
    position = (entries.cumsum() > exits.cumsum()).astype(int)

    # Align with original index and ensure integer dtype
    position = position.reindex(close.index).fillna(0).astype(int)

    return {"ohlcv": position}
