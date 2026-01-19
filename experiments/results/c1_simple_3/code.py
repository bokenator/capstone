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
        raise ValueError("data must contain 'ohlcv' key")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close: pd.Series = df["close"].astype(float)

    # Validate and extract params (only allowed params are used)
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    # Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use ewm with min_periods to avoid producing RSI too early
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    # Handle edge cases where avg_loss == 0 or avg_gain == 0
    # If avg_loss == 0 and avg_gain == 0 -> neutral 50
    both_zero = (avg_loss == 0) & (avg_gain == 0)
    only_loss_zero = (avg_loss == 0) & (~both_zero)
    only_gain_zero = (avg_gain == 0) & (~both_zero)

    rsi = rsi.copy()
    rsi[both_zero] = 50.0
    rsi[only_loss_zero] = 100.0
    rsi[only_gain_zero] = 0.0

    # Prepare signals: crosses below oversold -> entry, crosses above overbought -> exit
    prev_rsi = rsi.shift(1)

    entry_mask = (prev_rsi > oversold) & (rsi <= oversold)
    exit_mask = (prev_rsi < overbought) & (rsi >= overbought)

    # Ensure masks are False where RSI is NaN (warmup period)
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series: 1 for long, 0 for flat (long-only)
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    for idx in close.index:
        if not in_position:
            if entry_mask.loc[idx]:
                in_position = True
                position.loc[idx] = 1
            else:
                position.loc[idx] = 0
        else:
            # currently long
            if exit_mask.loc[idx]:
                in_position = False
                position.loc[idx] = 0
            else:
                position.loc[idx] = 1

    return {"ohlcv": position}
