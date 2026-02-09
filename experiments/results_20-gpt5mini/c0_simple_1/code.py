"""
RSI Mean Reversion Signal Generator

This module exports a single function `generate_signals` which computes RSI (14)
and generates long-only entry/exit position series based on RSI mean reversion
rules:

- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)

The function returns a dictionary with key "ohlcv" containing a pandas Series of
position targets: 1 for long, 0 for flat. The Series is aligned with the input
close prices index and contains no NaNs (filled with 0 where necessary).

The implementation uses Wilder's smoothing via EMA (alpha=1/period) for RSI
calculation and handles edge cases such as insufficient data and NaNs.

Author: VectorBT strategy generator
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close index. Initial values will be NaN until
        enough data is available (min_periods = period).
    """
    # Ensure close is a float series
    close = close.astype(float)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via EWM with alpha=1/period (adjust=False)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle divide-by-zero: if avg_loss == 0 => RSI = 100 (no losses),
    # if avg_gain == 0 and avg_loss > 0 => RSI = 0 (no gains).
    rsi = rsi.where(~(avg_loss == 0), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)

    # If both avg_gain and avg_loss are zero (flat price), set RSI to 50
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    if flat_mask.any():
        rsi = rsi.where(~flat_mask, 50.0)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy rules:
    - RSI period = 14
    - Enter long when RSI crosses below 30
    - Exit long when RSI crosses above 70

    Args:
        data: Dictionary containing input DataFrames. Must include the key
            "ohlcv" with a DataFrame that has a "close" column.
        params: Parameters dictionary (not used for this fixed-logic strategy,
            but accepted for API compatibility).

    Returns:
        A dict with a single key "ohlcv" whose value is a pandas Series of
        position targets (1 for long, 0 for flat), indexed the same as the
        input close prices.

    Raises:
        ValueError: if required data is missing or malformed.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].copy()
    if close.isnull().all():
        raise ValueError("close price series contains only NaNs")

    # Compute RSI
    period = 14
    rsi = _compute_rsi(close, period=period)

    # Define crossing conditions
    # Entry: RSI crosses below 30 (prev >= 30 and curr < 30)
    # Exit: RSI crosses above 70 (prev <= 70 and curr > 70)
    prev_rsi = rsi.shift(1)

    entry_mask = (prev_rsi >= 30) & (rsi < 30)
    exit_mask = (prev_rsi <= 70) & (rsi > 70)

    # Build position series iteratively to maintain state between bars
    position = pd.Series(0, index=rsi.index, dtype=int)
    state = 0  # 0 = flat, 1 = long

    # Iterate through bars in order
    for i in range(len(rsi)):
        # Use .iloc for positional access
        rsi_i = rsi.iloc[i]

        # Keep previous state if RSI is NaN (warmup period)
        if pd.isna(rsi_i):
            position.iloc[i] = state
            continue

        if state == 0:
            if entry_mask.iloc[i]:
                state = 1
        elif state == 1:
            if exit_mask.iloc[i]:
                state = 0

        position.iloc[i] = state

    # Ensure no NaNs and proper dtype
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
