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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Load and validate params (use only declared params)
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        raise TypeError("rsi_period must be an integer")
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    try:
        oversold = float(params.get("oversold", 30.0))
    except Exception:
        raise TypeError("oversold must be a float")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")

    try:
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise TypeError("overbought must be a float")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")

    if not (oversold < overbought):
        raise ValueError("oversold must be strictly less than overbought")

    # Handle empty or too-short series
    if close.shape[0] == 0:
        # Return empty position series with same index
        return {"ohlcv": pd.Series(dtype="int8", index=close.index)}

    if close.shape[0] <= rsi_period:
        # Not enough data to compute RSI -> always flat
        return {"ohlcv": pd.Series(0, index=close.index, dtype="int8")}

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use EWM with alpha=1/period and min_periods=period to respect warmup
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Prevent division by zero: where avg_loss == 0 -> set RSI to 100 (all gains)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # rsi will be NaN for the warmup period; keep NaNs so we don't generate signals on incomplete data

    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs in masks with False to avoid spurious signals
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series (long-only: 1 when in position, 0 when flat)
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = False
    # Iterate over index to maintain state machine for entries/exits
    for i in range(len(close)):
        if entry_mask.iat[i] and not in_position:
            in_position = True
        elif exit_mask.iat[i] and in_position:
            in_position = False
        position.iat[i] = 1 if in_position else 0

    return {"ohlcv": position}
