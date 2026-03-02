import pandas as pd
import numpy as np
from typing import Dict, Any


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
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = df["close"].astype(float)

    # Extract and validate parameters (only use keys defined in PARAM_SCHEMA)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold threshold must be less than overbought threshold")

    # --- RSI Calculation (Wilder's smoothing using EWM with adjust=False) ---
    # delta: price changes
    delta = close.diff()
    # Gains and losses
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's EMA (recursive) - adjust=False ensures no lookahead and consistent results
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Relative Strength and RSI
    # Handle division by zero carefully
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fix edge cases where roll_up or roll_down are zero
    rsi = rsi.copy()
    mask_up_zero = roll_up == 0
    mask_down_zero = roll_down == 0
    both_zero = mask_up_zero & mask_down_zero

    # If no gains and no losses -> neutral 50
    rsi.loc[both_zero] = 50.0
    # If no losses -> RSI = 100 (strong up)
    rsi.loc[mask_down_zero & ~mask_up_zero] = 100.0
    # If no gains -> RSI = 0 (strong down)
    rsi.loc[mask_up_zero & ~mask_down_zero] = 0.0

    # Any remaining NaNs (e.g., first bar) set to neutral 50
    rsi = rsi.fillna(50.0)

    # --- Signal generation: detect crossings ---
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (was >= oversold, now < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought (was <= overbought, now > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series aligned and no NaN
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # --- Build position series (long-only state machine) ---
    positions = np.zeros(len(close), dtype=int)
    pos = 0
    for i in range(len(close)):
        if entries.iat[i] and pos == 0:
            pos = 1
        elif exits.iat[i] and pos == 1:
            pos = 0
        positions[i] = pos

    position_series = pd.Series(positions, index=close.index, name="position").astype(int)

    return {"ohlcv": position_series}
