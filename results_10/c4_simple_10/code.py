import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy:
    - Compute RSI over `rsi_period` (Wilder smoothing using ewm alpha=1/period)
    - Enter LONG when RSI CROSSES BELOW `oversold` (i.e., previous RSI >= oversold and current RSI < oversold)
    - Exit LONG when RSI CROSSES ABOVE `overbought` (i.e., previous RSI <= overbought and current RSI > overbought)

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period (2-100)
            - oversold (float): RSI level to trigger entry when RSI CROSSES BELOW it (0-50)
            - overbought (float): RSI level to trigger exit when RSI CROSSES ABOVE it (50-100)

    Returns:
        Dict mapping slot names to position Series. Long-only strategy: values are 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping slot names to pandas DataFrames")

    if "ohlcv" not in data:
        raise KeyError("`data` must contain 'ohlcv' key with a DataFrame containing a 'close' column")

    df = data["ohlcv"]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("`data['ohlcv']` must contain a 'close' column")

    close = df["close"].astype(float)

    # Validate and extract params
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        raise ValueError("rsi_period must be an integer")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    try:
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise ValueError("oversold and overbought must be numbers")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")

    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use ewm with alpha=1/period and adjust=False for Wilder smoothing
    # min_periods ensures RSI is NaN until we have enough data
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0 (no losses), RSI should be 100 (overbought). If avg_gain is 0 (no gains), RSI should be 0.
    # Handle division by zero/infinite cases explicitly to avoid NaNs where sensible.
    rsi = rsi.where(~(avg_loss == 0) | (avg_gain == 0), 100.0)  # where loss==0 and gain>0 -> 100
    rsi = rsi.where(~(avg_gain == 0) | (avg_loss == 0), 0.0)    # where gain==0 and loss>0 -> 0
    # Note: if both avg_gain and avg_loss are 0, keep NaN (no price movement / insufficient data)

    # Detect crossings: previous RSI compared to threshold
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean Series (no NaNs)
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Construct position series: 1 for long, 0 for flat
    position = pd.Series(data=np.nan, index=close.index, name="position", dtype=float)

    # On entry days set to 1, on exit days set to 0. Then forward-fill to maintain position until exit.
    position.loc[entry_signals] = 1.0
    position.loc[exit_signals] = 0.0

    position = position.ffill().fillna(0.0)

    # Cast to integer positions (0 or 1)
    position = position.astype(int)

    return {"ohlcv": position}
