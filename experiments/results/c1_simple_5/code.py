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
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise ValueError("data['ohlcv'] DataFrame must contain 'close' column")

    close = df["close"].astype(float)

    # Validate params and use defaults if missing
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    # Extract parameters with validation according to PARAM_SCHEMA
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30.0)
    overbought = params.get("overbought", 70.0)

    # Type and range checks
    try:
        rsi_period = int(rsi_period)
    except Exception:
        raise ValueError("rsi_period must be an integer")
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period out of allowed range [2, 100]")

    try:
        oversold = float(oversold)
        overbought = float(overbought)
    except Exception:
        raise ValueError("oversold and overbought must be floats")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold out of allowed range [0.0, 50.0]")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought out of allowed range [50.0, 100.0]")

    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Calculate RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use exponential weighted mean as Wilder's smoothing
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Avoid division by zero
    rs = roll_up / roll_down
    # Where roll_down == 0 and roll_up == 0 => RSI undefined, set to 50 (neutral)
    rsi = pd.Series(np.nan, index=close.index)
    mask_both_zero = (roll_down == 0) & (roll_up == 0)
    mask_down_zero = (roll_down == 0) & (roll_up > 0)
    mask_up_zero = (roll_up == 0) & (roll_down > 0)

    rsi[mask_both_zero] = 50.0
    rsi[mask_down_zero] = 100.0
    rsi[mask_up_zero] = 0.0

    # For remaining values where roll_down > 0
    valid = ~(mask_both_zero | mask_down_zero | mask_up_zero)
    rsi.loc[valid] = 100.0 - (100.0 / (1.0 + rs.loc[valid]))

    # Ensure RSI is numeric and aligns with index
    rsi = rsi.astype(float)

    # Generate crossing signals
    # Entry: RSI crosses below oversold -> prev >= oversold and current < oversold
    prev_rsi = rsi.shift(1)
    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought -> prev <= overbought and current > overbought
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs with False to avoid accidental signals during warmup
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series: +1 for long, 0 for flat. Long-only strategy.
    position = pd.Series(0, index=close.index, dtype="int8")

    current_pos = 0
    for idx in close.index:
        if current_pos == 0:
            if entry_signal.loc[idx]:
                current_pos = 1
        elif current_pos == 1:
            if exit_signal.loc[idx]:
                current_pos = 0
        position.loc[idx] = current_pos

    return {"ohlcv": position}
