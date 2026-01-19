import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
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
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close: pd.Series = df["close"].astype(float)

    # Extract and validate parameters (only allowed params)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic parameter validation according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use min_periods=rsi_period to avoid early misleading values
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # rsi remains NaN where not enough data
    rsi = rsi

    # Detect cross events
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold (was >= oversold, now < oversold)
    entry_signal = (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (was <= overbought, now > overbought)
    exit_signal = (rsi_prev <= overbought) & (rsi > overbought)

    # Ensure signals are False where RSI is NaN
    entry_signal = entry_signal & rsi.notna() & rsi_prev.notna()
    exit_signal = exit_signal & rsi.notna() & rsi_prev.notna()

    # Build position series (0 = flat, 1 = long). Long-only strategy.
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_position = False

    # Iterate over index to create stateful position (handles warmup and sequential signals)
    for idx in close.index:
        if not in_position:
            if entry_signal.loc[idx]:
                in_position = True
        else:
            if exit_signal.loc[idx]:
                in_position = False

        pos.loc[idx] = 1 if in_position else 0

    # Return positions for the ohlcv slot. No short positions generated.
    return {"ohlcv": pos}
