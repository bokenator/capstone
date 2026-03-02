import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series
        period: RSI period (int)

    Returns:
        RSI as a pandas Series with the same index as close.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: use ewm with alpha=1/period
    # Require at least `period` observations to start producing values
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    This is a long-only strategy that goes long when RSI crosses below the
    `oversold` threshold and exits when RSI crosses above the `overbought` threshold.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with
              a DataFrame that has a 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Position values are 1 (long) or 0 (flat).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = df["close"].astype(float)

    # Validate and read params (use only allowed params)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Param sanity checks according to PARAM_SCHEMA
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings:
    # - Entry when RSI crosses below `oversold` (prev >= oversold and current < oversold)
    # - Exit when RSI crosses above `overbought` (prev <= overbought and current > overbought)
    prev_rsi = rsi.shift(1)

    cross_below = prev_rsi.notna() & (prev_rsi >= oversold) & (rsi < oversold)
    cross_above = prev_rsi.notna() & (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype="int8")

    in_long = False
    # Iterate in chronological order to ensure proper stateful behavior
    for ts in close.index:
        if in_long:
            # If currently long, exit if cross_above
            if bool(cross_above.loc[ts]):
                in_long = False
                position.loc[ts] = 0
            else:
                position.loc[ts] = 1
        else:
            # If currently flat, enter if cross_below
            if bool(cross_below.loc[ts]):
                in_long = True
                position.loc[ts] = 1
            else:
                position.loc[ts] = 0

    # Ensure no NaNs and integer dtype
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
