"""
RSI Mean Reversion Signal Generator

Implements an RSI-based mean reversion strategy:
- Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
- Enter long when RSI crosses below `oversold` (e.g., 30)
- Exit long when RSI crosses above `overbought` (e.g., 70)

Function exported:
- generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

This module purposely avoids using vectorbt inside signal generation so it can be
used with any backtesting framework.
"""

from __future__ import annotations

import typing
import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (exponential moving average).

    Args:
        close: Close price series.
        period: RSI period (must be >= 2).

    Returns:
        RSI values as a pandas Series with the same index as `close`.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing via ewm with alpha=1/period
    # Set min_periods=period so initial RSI values are NaN until enough data
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # Handle division by zero: when avg_loss == 0, rs -> +inf -> rsi -> 100
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Ensure same index and name
    rsi.name = "rsi"
    rsi.index = close.index
    return rsi


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
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
    # Validate input data structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Extract and validate params
    # Allowed params: rsi_period, oversold, overbought
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic bounds checking according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Prepare crossing conditions
    # Entry: RSI crosses BELOW oversold (prev >= oversold and curr < oversold)
    prev_rsi = rsi.shift(1)

    entry_cond = (
        prev_rsi.notna()
        & rsi.notna()
        & (prev_rsi >= oversold)
        & (rsi < oversold)
    )

    # Exit: RSI crosses ABOVE overbought (prev <= overbought and curr > overbought)
    exit_cond = (
        prev_rsi.notna()
        & rsi.notna()
        & (prev_rsi <= overbought)
        & (rsi > overbought)
    )

    # Build the position Series using a stateful forward-fill approach
    # Create a series of signals where entries set 1, exits set 0, others NaN
    signals = pd.Series(index=close.index, dtype="float64")
    signals.loc[entry_cond] = 1.0
    signals.loc[exit_cond] = 0.0

    # Forward-fill signals and default to 0 (flat) if no prior signal
    position = signals.ffill().fillna(0.0).astype(int)

    # Ensure position values are only 0 or 1
    position = position.where(position.isin([0, 1]), 0).astype(int)

    return {"ohlcv": position}
