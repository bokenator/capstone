"""
RSI Mean Reversion Signal Generator

Implements the generate_signals function required by the prompt.

Strategy:
- Compute RSI (Wilder's smoothing / EWM) with given period
- Enter long when RSI crosses below oversold threshold
- Exit long when RSI crosses above overbought threshold
- Long-only, returns position series with values 0 or 1

Only accesses 'close' column from data['ohlcv'] as required by DATA_SCHEMA.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI period (int >= 2).

    Returns:
        RSI series indexed like `close`. Values are NaN until enough data is available.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via EWM (alpha = 1/period). Use min_periods=period so
    # RSI is NaN until we have `period` observations.
    roll_up = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative Strength
    rs = roll_up / roll_down

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Where both roll_up and roll_down are 0 (no price movement), define RSI=50
    both_zero = (roll_up == 0) & (roll_down == 0)
    rsi = rsi.mask(both_zero, 50.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY: 1 (long) or 0 (flat).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = df["close"].astype(float)

    # Extract and validate params
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold level must be strictly less than overbought level")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Prepare previous RSI for cross detection
    rsi_prev = rsi.shift(1)

    # Detect crossing events. Require both current and previous RSI to be finite to avoid
    # issuing signals during warmup (NaN) period.
    valid_prev = rsi_prev.notna()
    valid_curr = rsi.notna()

    # Entry: RSI crosses BELOW the oversold threshold (from >= to <)
    entries = (valid_prev & valid_curr) & (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE the overbought threshold (from <= to >)
    exits = (valid_prev & valid_curr) & (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (0/1) by scanning through time
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = False
    # Use integer location based iteration for performance on large Series
    for i in range(len(position)):
        if not in_position:
            # If not in position and entry signal fires, enter
            if entries.iloc[i]:
                in_position = True
        else:
            # If in position and exit signal fires, exit
            if exits.iloc[i]:
                in_position = False
        position.iloc[i] = 1 if in_position else 0

    return {"ohlcv": position}
