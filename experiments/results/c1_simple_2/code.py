import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict
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
        raise KeyError("data must contain 'ohlcv' key")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Extract and validate params
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Calculate RSI using Wilder's smoothing (EMA with alpha=1/period)
    # Handle NaNs in close
    close_clean = close.copy()

    # Compute price changes
    delta = close_clean.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted moving average with alpha=1/rsi_period
    # For the first value, we will have NaNs until enough data exists
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Prevent division by zero
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss is zero (no losses), RSI should be 100; where avg_gain is zero (no gains), RSI should be 0
    rsi = rsi.fillna(0.0)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0  # flat price -> neutral

    # Build signals: entry when RSI crosses below oversold; exit when RSI crosses above overbought
    rsi_prev = rsi.shift(1)

    entry_signals = (rsi_prev >= oversold) & (rsi < oversold)
    exit_signals = (rsi_prev <= overbought) & (rsi > overbought)

    # Initialize position series
    position = pd.Series(data=0, index=close.index, dtype=int)

    in_position = False

    # Iterate through index to build position path
    for idx in range(len(position)):
        if idx == 0:
            position.iloc[idx] = 0
            continue

        if not in_position:
            # Check for entry
            if entry_signals.iloc[idx]:
                in_position = True
                position.iloc[idx] = 1
            else:
                position.iloc[idx] = 0
        else:
            # Currently in position; check for exit
            if exit_signals.iloc[idx]:
                in_position = False
                position.iloc[idx] = 0
            else:
                position.iloc[idx] = 1

    # Ensure positions are 0 before RSI is valid (warmup period)
    rsi_valid = rsi.notna()
    position[~rsi_valid] = 0

    return {"ohlcv": position}
