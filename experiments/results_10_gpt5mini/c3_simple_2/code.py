"""
RSI Mean Reversion Signal Generator

Implements a long-only mean reversion strategy based on RSI:
- RSI period configurable (default 14)
- Go long when RSI crosses below the oversold threshold
- Exit when RSI crosses above the overbought threshold

Function:
    generate_signals(data: dict, params: dict) -> dict

Returns:
    {"ohlcv": position_series} where position_series contains 0 (flat) or 1 (long)

The implementation is careful to avoid lookahead bias and handles edge cases.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI period (integer > 0).

    Returns:
        pd.Series with RSI values indexed the same as `close`.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    close = close.astype(float)
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via ewm with alpha=1/period
    # min_periods=period ensures RSI is NaN until we have enough data
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long (1) / flat (0) position signals based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" mapping to a DataFrame that contains a "close" column.
        params: dict with keys:
            - 'rsi_period' (int): RSI period, e.g., 14
            - 'oversold' (float): oversold threshold, e.g., 30.0
            - 'overbought' (float): overbought threshold, e.g., 70.0

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series of 0/1
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with key 'ohlcv'")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Read params with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Prepare output series (default flat)
    n = len(close)
    position = pd.Series(0, index=close.index, dtype="int8")

    if n == 0:
        return {"ohlcv": position}

    # Compute RSI (only depends on past data)
    rsi = _compute_rsi(close, rsi_period)

    # Entry: RSI crosses below oversold (previous >= oversold and current < oversold)
    prev_rsi = rsi.shift(1)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (previous <= overbought and current > overbought)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # State machine to build position series (ensures no double entries/exits)
    in_position = False
    # Use integer positions 0/1, deterministic and no lookahead (only uses current/previous RSI)
    for i in range(n):
        # index accessor
        idx = close.index[i]

        if in_position:
            # If exit signal, leave position
            if bool(exit_signals.iloc[i]):
                in_position = False
        else:
            # If entry signal and currently flat, enter
            if bool(entry_signals.iloc[i]):
                in_position = True

        position.iloc[i] = 1 if in_position else 0

    # Ensure no NaNs (fill with 0 just in case) and restrict values to {0,1}
    position = position.fillna(0).astype("int8")

    return {"ohlcv": position}
