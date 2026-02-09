"""
RSI Mean Reversion Signal Generator

Implements generate_signals as specified in the prompt.

- Computes RSI using Wilder's smoothing (EWMA with alpha=1/period)
- Entry when RSI crosses below `oversold`
- Exit when RSI crosses above `overbought`
- Long-only, single-asset

The function is defensive: it accepts either a dict with key 'ohlcv' or a plain
DataFrame (to support different test harness usage). It returns a dict with key
'ohlcv' mapping to a pd.Series of 0/1 positions.
"""

from typing import Dict

import numpy as np
import pandas as pd


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
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """

    # Accept either a dict with 'ohlcv' or a raw DataFrame (some tests may pass DF)
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
        df = data["ohlcv"]
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data must be a dict[str, DataFrame] or a DataFrame")

    if "close" not in df.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    # Copy to avoid mutating input
    close = df["close"].astype(float).copy()

    # Validate and extract parameters (only use keys declared in PARAM_SCHEMA)
    # Defaults are provided to be robust if params is missing keys
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic parameter validation (within allowed ranges)
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    # This implementation uses only past data (no lookahead)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # EWMA smoothing (adjust=False gives the recursive Wilder-like smoothing)
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Compute RSI, handling division-by-zero
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle cases where avg_loss == 0 (no losses): RSI should be 100 if gains exist,
    # or 50 if both gains and losses are zero (flat price)
    mask_loss_zero = (avg_loss == 0)
    if mask_loss_zero.any():
        # Where loss == 0 and gain > 0 -> RSI = 100
        mask_gain_pos = mask_loss_zero & (avg_gain > 0)
        rsi.loc[mask_gain_pos] = 100.0
        # Where both gain and loss are 0 -> RSI = 50 (neutral)
        mask_both_zero = mask_loss_zero & (avg_gain == 0)
        rsi.loc[mask_both_zero] = 50.0

    # Fill any remaining NaNs (e.g., first row) with neutral 50 to avoid NaNs in output
    rsi = rsi.fillna(50.0)

    # Detect crossings: entry when RSI crosses below oversold; exit when crosses above overbought
    prev_rsi = rsi.shift(1).fillna(50.0)

    entries = (rsi < oversold) & (prev_rsi >= oversold)
    exits = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series iteratively to ensure no double-entries and clean transitions
    pos = pd.Series(0, index=close.index, dtype="int8")

    in_long = 0
    # Use .iloc for positional access which works both for RangeIndex and DatetimeIndex
    for i in range(len(close)):
        # If flat and an entry signal occurs at this index -> go long
        if not in_long and entries.iloc[i]:
            in_long = 1
        # If long and an exit signal occurs at this index -> go flat
        elif in_long and exits.iloc[i]:
            in_long = 0
        pos.iloc[i] = in_long

    # Ensure output only contains 0 or 1 and has same index/length as input
    pos = pos.astype(int)

    return {"ohlcv": pos}
