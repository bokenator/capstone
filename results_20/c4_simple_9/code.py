import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute Wilder's RSI using exponential smoothing (alpha=1/period).

    Args:
        close: Close price series.
        period: RSI period (e.g., 14).

    Returns:
        RSI series (float) with same index as close. Initial periods are NaN.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing using EWM with alpha = 1/period
    # min_periods=period to ensure warmup
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases where avg_loss or avg_gain are zero
    # If avg_loss == 0 and avg_gain == 0 -> RSI = 50 (no movement)
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100 (all gains)
    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0 (all losses)
    loss_zero = avg_loss == 0
    gain_zero = avg_gain == 0

    # Create a copy to avoid SettingWithCopyWarning
    rsi = rsi.copy()

    # Only set values where we have enough data (min_periods satisfied)
    has_warmup = avg_gain.notna() | avg_loss.notna()

    rsi.loc[loss_zero & gain_zero & has_warmup] = 50.0
    rsi.loc[loss_zero & ~gain_zero & has_warmup] = 100.0
    rsi.loc[~loss_zero & gain_zero & has_warmup] = 0.0

    return rsi


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
        Dict mapping slot names to position Series (1 for long, 0 for flat).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = df["close"].astype(float)

    # Extract and validate params
    # Use PARAM_SCHEMA constraints
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params types: {e}")

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Determine cross below (entry) and cross above (exit)
    prev_rsi = rsi.shift(1)

    # Only consider crossings where both prev and current RSI are finite
    valid = prev_rsi.notna() & rsi.notna()

    entries = (prev_rsi >= oversold) & (rsi < oversold) & valid
    exits = (prev_rsi <= overbought) & (rsi > overbought) & valid

    # Build position series (long-only: 1 or 0)
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate over index to correctly handle sequential entries/exits
    for i, idx in enumerate(close.index):
        if entries.iloc[i] and not in_position:
            in_position = True
        elif exits.iloc[i] and in_position:
            in_position = False
        position.iloc[i] = 1 if in_position else 0

    return {"ohlcv": position}
