import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Signals are LONG-ONLY: 1 = long, 0 = flat.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with a DataFrame that has a 'close' column.
        params: Dict with strategy parameters:
            - rsi_period (int): RSI calculation period (2-100)
            - oversold (float): Enter long when RSI CROSSES BELOW this value (0-50)
            - overbought (float): Exit long when RSI CROSSES ABOVE this value (50-100)

    Returns:
        Dict mapping slot names to position Series. Example: {"ohlcv": pd.Series(...) }
    """

    # --- Validate input data ---
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise ValueError("`data` must contain an 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column as specified in DATA_SCHEMA")

    close: pd.Series = ohlcv["close"].astype(float).copy()

    # Ensure index is preserved
    index = close.index

    # --- Validate and extract parameters (only use allowed params) ---
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        raise ValueError("rsi_period must be an integer")
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    try:
        oversold = float(params.get("oversold", 30.0))
    except Exception:
        raise ValueError("oversold must be a float")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")

    try:
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise ValueError("overbought must be a float")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")

    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought for meaningful signals")

    # --- Compute RSI (Wilder's smoothing via EWM) ---
    # Delta between consecutive closes
    delta = close.diff()

    # Gains (positive deltas) and losses (positive values)
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Use Wilder's smoothing: exponential moving average with alpha = 1/period
    # adjust=False to match the recursive form
    avg_gain = gains.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # RSI = 100 * avg_gain / (avg_gain + avg_loss)
    denom = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 * (avg_gain / denom)

    # Where denom == 0 (no movement), set RSI to 50 (neutral)
    rsi = rsi.where(denom != 0.0, 50.0)

    # Keep same index
    rsi = pd.Series(rsi, index=index, name="rsi")

    # --- Generate entry/exit signals based on RSI crosses ---
    # Entry: RSI crosses BELOW oversold (from >= oversold -> < oversold)
    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)

    # Exit: RSI crosses ABOVE overbought (from <= overbought -> > overbought)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)

    # Replace possible NA booleans with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # --- Build position series (long-only) ---
    pos_vals = np.zeros(len(close), dtype=np.int8)
    in_long = False

    # Use integer position iteration for clarity and correctness on edge cases
    for i in range(len(close)):
        # If not currently long and an entry signal occurs -> go long
        if (not in_long) and bool(entries.iat[i]):
            in_long = True
        # If currently long and an exit signal occurs -> exit to flat
        elif in_long and bool(exits.iat[i]):
            in_long = False

        pos_vals[i] = 1 if in_long else 0

    position = pd.Series(pos_vals, index=index, name="position").astype("int8")

    return {"ohlcv": position}
