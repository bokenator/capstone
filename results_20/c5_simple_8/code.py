# RSI mean reversion signal generator
# Implements generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI period >= 2.

    Returns:
        RSI series (float), same index as close. Initial `period` values are NaN.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Delta
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing using EWM with alpha = 1/period (adjust=False)
    # This uses only past information (no lookahead)
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle cases where avg_loss is zero -> rsi -> 100
    rsi = rsi.where(~(avg_loss == 0.0), 100.0)
    rsi = rsi.where(~(avg_gain == 0.0), 0.0).astype(float)

    # Enforce NaN for the warmup period to be consistent
    if len(rsi) >= period:
        rsi.iloc[:period] = np.nan
    else:
        rsi[:] = np.nan

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
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
    """
    # Accept either a dict with 'ohlcv' or a direct DataFrame/Series for convenience
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        df = pd.DataFrame(data) if isinstance(data, pd.Series) else data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("`data` dict must contain 'ohlcv' key with a DataFrame containing 'close' column")
        df = data["ohlcv"]
    else:
        raise TypeError("`data` must be a dict with 'ohlcv' key or a pandas DataFrame/Series")

    if "close" not in df.columns:
        # If a bare single-column DataFrame was passed, treat the first column as close
        if df.shape[1] == 1:
            close = df.iloc[:, 0]
        else:
            raise ValueError("Input data must contain 'close' column")
    else:
        close = df["close"]

    # Parameters with validation and defaults
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    rsi_period = max(2, min(100, rsi_period))

    oversold = float(params.get("oversold", 30.0)) if params is not None else 30.0
    oversold = max(0.0, min(50.0, oversold))

    overbought = float(params.get("overbought", 70.0)) if params is not None else 70.0
    overbought = max(50.0, min(100.0, overbought))

    if oversold >= overbought:
        raise ValueError("`oversold` must be strictly less than `overbought`")

    # Compute RSI (only uses past data => no lookahead)
    rsi = _compute_rsi(close, rsi_period)

    # Entry: go long when RSI is below oversold (mean-reversion buy)
    # Exit: close long when RSI is above overbought
    # Use current bar values (no future data). Require RSI to be non-NaN.
    can_enter = rsi < oversold
    can_exit = rsi > overbought

    # Build position series by scanning forward to avoid double entries/exits
    position = pd.Series(0, index=close.index, dtype="int8")
    in_position = False

    for i in range(len(position)):
        # Skip bars where RSI is NaN (warmup)
        if pd.isna(rsi.iat[i]):
            position.iat[i] = 1 if in_position else 0
            continue

        if (not in_position) and can_enter.iat[i]:
            in_position = True
        elif in_position and can_exit.iat[i]:
            in_position = False

        position.iat[i] = 1 if in_position else 0

    # Ensure no NaNs in the output
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
