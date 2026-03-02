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
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close: pd.Series = ohlcv["close"].astype(float)

    # Read and validate parameters (only allowed params are accessed)
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if rsi_period > 100:
        raise ValueError("rsi_period must be <= 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be within [0, 50]")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be within [50, 100]")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing approximation (EMA with alpha = 1/period)
    # This implementation only uses the 'close' series (per DATA_SCHEMA)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use EWM to approximate Wilder smoothing
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Set warmup period values to NaN to avoid spurious signals before RSI is meaningful
    rsi.iloc[:rsi_period] = np.nan

    # Detect crossings
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entry_signal = (rsi < oversold) & (rsi_prev >= oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exit_signal = (rsi > overbought) & (rsi_prev <= overbought)

    # Build position series (long-only: 1 = long, 0 = flat)
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate chronologically to maintain position until exit
    for i in range(len(position)):
        if not in_position:
            if entry_signal.iat[i]:
                in_position = True
                position.iat[i] = 1
            else:
                position.iat[i] = 0
        else:
            # currently long
            if exit_signal.iat[i]:
                in_position = False
                position.iat[i] = 0
            else:
                position.iat[i] = 1

    return {"ohlcv": position}
