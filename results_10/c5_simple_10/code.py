"""
RSI Mean Reversion Signal Generator

Implements generate_signals(...) as required by the prompt. Computes RSI using Wilder's smoothing
and produces long-only position signals: enter when RSI crosses below oversold, exit when RSI
crosses above overbought.

Only uses 'close' column from the provided data and only the parameters declared in PARAM_SCHEMA.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI period (int).

    Returns:
        RSI series aligned with close.index. Initial values before warmup are NaN.
    """
    # Validate inputs
    if period < 1:
        raise ValueError("rsi_period must be >= 1")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing using ewm with alpha=1/period, adjust=False
    # min_periods=period ensures we only compute RSI after 'period' values
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Calculate RSI, handle division by zero
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss == 0 -> RSI = 100 (all gains). If avg_gain == 0 -> RSI = 0 (all losses).
    # The formula above already yields these values via inf/0 handling, but ensure NaNs
    # (where both avg_gain and avg_loss are 0) are set to 50
    rsi = rsi.where(~(avg_gain.isna() & avg_loss.isna()), other=np.nan)

    # Keep as pandas Series
    rsi = pd.Series(rsi, index=close.index)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              or it can be passed directly as a DataFrame (backwards compatibility).
              The DataFrame must have a 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. This is a LONG-ONLY strategy,
        so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Accept either a dict with 'ohlcv' or a DataFrame directly for flexibility
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("data must contain 'ohlcv' key with a DataFrame")
        df = data["ohlcv"]
    else:
        raise TypeError("data must be a dict with key 'ohlcv' or a pandas DataFrame")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    # Extract parameters with defaults (but only use allowed keys)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period out of allowed bounds")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold out of allowed bounds")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought out of allowed bounds")

    close = df["close"].astype(float).copy()

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Determine crossing signals (use previous bar to detect cross)
    prev_rsi = rsi.shift(1)

    entry_signal = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signal = (rsi > overbought) & (prev_rsi <= overbought)

    # Ensure boolean Series aligned
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series (0 = flat, 1 = long) using a simple state machine
    position = pd.Series(0, index=close.index, dtype="int8")
    in_pos = False

    # Use integer-based indexing for speed and determinism
    for i in range(len(position)):
        if entry_signal.iat[i] and not in_pos:
            in_pos = True
            position.iat[i] = 1
        elif exit_signal.iat[i] and in_pos:
            in_pos = False
            position.iat[i] = 0
        else:
            # carry forward previous state
            position.iat[i] = 1 if in_pos else 0

    # Replace any potential NaNs with 0 (shouldn't be any but be defensive)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
