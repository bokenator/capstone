import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy:
    - Calculate RSI using Wilder's smoothing (EMA with alpha=1/period)
    - Go long when RSI crosses below `oversold` (e.g., 30)
    - Exit long when RSI crosses above `overbought` (e.g., 70)
    - Long-only: positions are 1 (long) or 0 (flat)

    Args:
        data: Dict containing an 'ohlcv' DataFrame with a 'close' column.
        params: Dict with keys:
            - rsi_period (int): RSI period (2-100)
            - oversold (float): entry threshold (0-50)
            - overbought (float): exit threshold (50-100)

    Returns:
        Dict mapping slot name 'ohlcv' to a pd.Series of positions (0 or 1).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("`data` must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("`data['ohlcv']` must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("`ohlcv` DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Validate and extract parameters (only allowed keys)
    if not isinstance(params, dict):
        raise ValueError("`params` must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Handle trivial case of empty series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int64")}

    # Compute RSI using Wilder's smoothing (EMA with alpha = 1 / period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's EMA (RMA)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Raw RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss or avg_gain are zero
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    mask_loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
    if mask_loss_zero_gain_pos.any():
        rsi.loc[mask_loss_zero_gain_pos] = 100.0

    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    mask_gain_zero_loss_pos = (avg_gain == 0) & (avg_loss > 0)
    if mask_gain_zero_loss_pos.any():
        rsi.loc[mask_gain_zero_loss_pos] = 0.0

    # If both are zero (flat prices) -> RSI = 50 (neutral)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    if mask_both_zero.any():
        rsi.loc[mask_both_zero] = 50.0

    # Ensure rsi aligns with index
    rsi = rsi.reindex(close.index)

    # Detect cross below oversold for entries and cross above overbought for exits
    prev_rsi = rsi.shift(1)

    entries = (prev_rsi > oversold) & (rsi <= oversold)
    exits = (prev_rsi < overbought) & (rsi >= overbought)

    # Replace NaNs with False (no signal during warmup)
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series (0 = flat, 1 = long)
    pos_array = np.zeros(len(close), dtype=np.int8)
    in_position = 0

    # Use .iat for performance and to avoid SettingWithCopy
    for i in range(len(close)):
        if entries.iat[i] and in_position == 0:
            in_position = 1
        elif exits.iat[i] and in_position == 1:
            in_position = 0
        pos_array[i] = in_position

    positions = pd.Series(pos_array, index=close.index, name="position")

    return {"ohlcv": positions}
