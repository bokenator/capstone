"""
RSI Mean Reversion Signal Generator

Generates long-only position signals based on RSI crossings:
- Go long when RSI crosses below `oversold` level
- Exit when RSI crosses above `overbought` level

Only accesses `data['ohlcv']['close']` per DATA_SCHEMA and only uses
parameters declared in PARAM_SCHEMA.
"""

from typing import Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Wilder's RSI for a price series.

    Uses exponential smoothing with alpha=1/period (Wilder's smoothing).
    Returns a Series aligned with `close` index. Values will be NaN for
    the warmup period.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: use ewm with alpha=1/period. Require at least `period` observations
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero; where avg_loss == 0 set RSI to 100 (no losses),
    # where both gains and losses are zero set RSI to 50 (flat price)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle special cases
    rsi = pd.Series(rsi, index=close.index)

    # avg_loss == 0 -> RSI = 100 (only gains)
    rsi = rsi.where(avg_loss != 0.0, 100.0)

    # avg_gain == 0 and avg_loss == 0 -> RSI = 50 (no movement)
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.where(~both_zero, 50.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict,
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
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot name to DataFrame")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].copy()

    # Validate and extract parameters (use only declared params)
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        raise ValueError("rsi_period must be an integer")
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period out of allowed range [2, 100]")

    try:
        oversold = float(params.get("oversold", 30.0))
    except Exception:
        raise ValueError("oversold must be a float")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold out of allowed range [0.0, 50.0]")

    try:
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise ValueError("overbought must be a float")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought out of allowed range [50.0, 100.0]")

    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings:
    # Entry: RSI crosses below oversold: previous >= oversold and current < oversold
    # Exit: RSI crosses above overbought: previous <= overbought and current > overbought
    prev_rsi = rsi.shift(1)

    entries = (prev_rsi >= oversold) & (rsi < oversold)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN with False for signals
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build target series: 1 on entry points, 0 on exit points, NaN elsewhere
    target = pd.Series(data=np.nan, index=close.index, dtype=float)
    if entries.any():
        target.loc[entries] = 1.0
    if exits.any():
        target.loc[exits] = 0.0

    # Forward fill the targets to produce position, default to 0 (flat)
    position = target.ffill().fillna(0.0)

    # Ensure long-only discrete positions {0,1}
    # Round and clip to avoid numerical artifacts, then cast to int
    position = position.round().clip(lower=0.0, upper=1.0).astype(int)

    return {"ohlcv": position}
