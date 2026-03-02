# RSI Mean Reversion Signal Generator
# Implements generate_signals for a long-only RSI mean reversion strategy

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

    # Validate input data structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = df["close"].astype(float).copy()

    # Validate and extract params (only allowed params are used)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation against provided PARAM_SCHEMA
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI using Wilder's smoothing (EMA with alpha = 1/period)
    # Handle NaNs in close gracefully
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use EWM with adjust=False for Wilder's smoothing
    # The first values will be NaN until enough data is present
    avg_gain = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RSI, handle divide-by-zero
    rs = avg_gain / avg_loss
    # When avg_loss == 0 and avg_gain == 0 -> RSI undefined; set to 50 as neutral
    rsi = pd.Series(index=close.index, dtype=float)
    mask_loss_zero = avg_loss == 0
    mask_gain_zero = avg_gain == 0

    # Where both zero -> neutral 50
    both_zero = mask_loss_zero & mask_gain_zero
    rsi[both_zero] = 50.0

    # Where loss == 0 but gain > 0 -> RSI = 100
    rsi[mask_loss_zero & ~mask_gain_zero] = 100.0

    # Where gain == 0 and loss > 0 -> RSI = 0
    rsi[mask_gain_zero & ~mask_loss_zero] = 0.0

    # Else compute normally
    normal_mask = ~(both_zero | mask_loss_zero | mask_gain_zero)
    rsi[normal_mask] = 100.0 - (100.0 / (1.0 + rs[normal_mask]))

    # Ensure RSI has same index and NaNs preserved where necessary
    rsi = rsi.reindex(close.index)

    # Generate entry (cross below oversold) and exit (cross above overbought) signals
    prev_rsi = rsi.shift(1)

    entry = (prev_rsi.notna()) & (prev_rsi >= oversold) & (rsi < oversold)
    exit = (prev_rsi.notna()) & (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series: 1 when long, 0 when flat
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate chronologically
    for i in range(len(close)):
        if not in_position and entry.iat[i]:
            in_position = True
            position.iat[i] = 1
        elif in_position:
            # If currently in position, set position to 1
            position.iat[i] = 1
            # Check for exit at this bar
            if exit.iat[i]:
                in_position = False
                position.iat[i] = 0
        # else remains 0

    return {"ohlcv": position}
