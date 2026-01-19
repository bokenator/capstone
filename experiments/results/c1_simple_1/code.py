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
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
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

    # Extract parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation of parameters
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be within [0.0, 50.0]")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be within [50.0, 100.0]")
    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
    # Handle NaNs in price series by propagating them through RSI
    delta = close.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use Wilder's smoothing: exponential moving average with alpha=1/rsi_period
    # adjust=False implements the recursive form
    ma_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Avoid division by zero
    rs = ma_up / ma_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where both ma_up and ma_down are zero (no moves), set RSI to 50
    zero_mask = (ma_up == 0) & (ma_down == 0)
    rsi = rsi.where(~zero_mask, 50.0)

    # Ensure RSI has same index as close
    rsi = rsi.reindex(close.index)

    # Generate entry/exit masks based on crosses
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Initialize position series (0 = flat, 1 = long)
    pos = pd.Series(0, index=close.index, dtype=int)

    in_long = False

    # Iterate over index to build position while respecting NaNs
    for i in range(len(close.index)):
        idx = close.index[i]

        # If price is NaN or RSI is NaN, remain flat (no position change)
        if pd.isna(close.iloc[i]) or pd.isna(rsi.iloc[i]):
            pos.iloc[i] = 0 if not in_long else 1
            continue

        if not in_long:
            if bool(entry_mask.iloc[i]):
                in_long = True
                pos.iloc[i] = 1
            else:
                pos.iloc[i] = 0
        else:
            # currently long
            if bool(exit_mask.iloc[i]):
                in_long = False
                pos.iloc[i] = 0
            else:
                pos.iloc[i] = 1

    # Return positions for the 'ohlcv' slot; no short positions used
    return {"ohlcv": pos}
