"""
RSI Mean Reversion Signal Generator

Implements generate_signals(...) for a long-only RSI mean reversion strategy.

Strategy:
- Compute RSI with Wilder smoothing (EWMA with alpha=1/period, min_periods=period)
- Enter long when RSI crosses BELOW the oversold threshold (e.g., 30)
- Exit long when RSI crosses ABOVE the overbought threshold (e.g., 70)

The function follows the signature required by the backtest runner.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        close: Series of close prices.
        period: RSI period (lookback).

    Returns:
        RSI as a pandas Series indexed like `close`.

    Notes:
        - Uses exponential weighted mean with alpha=1/period and min_periods=period
          to approximate Wilder's moving average. The first `period` values will be NaN.
        - Handles edge cases where average loss is zero.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via EWM with alpha=1/period
    # min_periods=period ensures NaN values for the warmup period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # Compute RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 and avg_gain == 0 -> price unchanged -> RSI = 50
    # - If avg_loss == 0 and avg_gain > 0 -> RSI = 100 (max)
    # - If avg_gain == 0 and avg_loss > 0 -> RSI = 0 (min)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
    gain_zero_loss_pos = (avg_gain == 0) & (avg_loss > 0)

    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(loss_zero_gain_pos, 100.0)
    rsi = rsi.mask(gain_zero_loss_pos, 0.0)

    # Ensure same index and name
    rsi.name = "rsi"
    rsi.index = close.index

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict,
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
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Validate parameters and use defaults from schema if missing
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Extract parameters with validation
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Define signals based on RSI crossings
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses BELOW oversold (from >= oversold to < oversold)
    entry_signal = (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE overbought (from <= overbought to > overbought)
    exit_signal = (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (long-only). Use an iterative approach to ensure clarity
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = False
    # Iterate using index to preserve non-integer indices (e.g., DatetimeIndex)
    for idx in close.index:
        if entry_signal.loc[idx] and not in_position:
            in_position = True
        if exit_signal.loc[idx] and in_position:
            in_position = False
        position.loc[idx] = 1 if in_position else 0

    # Ensure no NaNs and integer dtype
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
