import pandas as pd
import numpy as np
from typing import Dict


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

    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    # Extract close prices
    close = df["close"].astype(float).copy()

    # Read and validate params (use only allowed params)
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        raise ValueError("rsi_period must be an integer")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period out of allowed range [2, 100]")

    try:
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise ValueError("oversold and overbought must be numeric")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be within [0.0, 50.0]")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be within [50.0, 100.0]")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Handle trivial case of empty series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int64")}

    # Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average with alpha=1/period (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute raw RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_gain or avg_loss are zero
    # If both are zero (flat prices), set RSI to 50
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    mask_loss_zero = (avg_loss == 0) & (~mask_both_zero)
    mask_gain_zero = (avg_gain == 0) & (~mask_both_zero)

    rsi = rsi.copy()
    rsi.loc[mask_both_zero] = 50.0
    rsi.loc[mask_loss_zero] = 100.0
    rsi.loc[mask_gain_zero] = 0.0

    # Determine crossing events
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (was >= oversold, now < oversold)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (was <= overbought, now > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean dtype and align index
    entry_mask = entry_mask.fillna(False).astype(bool)
    exit_mask = exit_mask.fillna(False).astype(bool)

    # Build position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = 0
    # Iterate over bars to maintain stateful long-only position
    for i in range(len(close)):
        if pd.isna(rsi.iloc[i]):
            # No valid RSI yet -> remain flat
            position.iloc[i] = 0
            continue

        if in_position == 0:
            # Not in position, check for entry
            if entry_mask.iloc[i]:
                in_position = 1
        else:
            # In position, check for exit
            if exit_mask.iloc[i]:
                in_position = 0

        position.iloc[i] = int(in_position)

    # Return positions for the 'ohlcv' slot
    return {"ohlcv": position.astype(int)}
