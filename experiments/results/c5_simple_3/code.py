# Generated RSI mean reversion signal generator
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


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
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Validate input type
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot name to DataFrame (got %s)" % type(data))

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    # Extract parameters with validation
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    # Work on a copy to avoid mutating input
    close = df["close"].astype(float).copy()

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential moving average with alpha = 1/period -> equivalent to Wilder's smoothing
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss is zero and avg_gain is also zero, define RSI as 50
    rsi = rsi.fillna(50.0)

    # Signals: cross below oversold -> entry; cross above overbought -> exit
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean Series aligned with close index and no NaNs in signals
    entry_signal = entry_signal.reindex(close.index, copy=False).fillna(False)
    exit_signal = exit_signal.reindex(close.index, copy=False).fillna(False)

    # Build position series by simulating state machine to avoid double entries
    pos = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    # Iterate over index to maintain determinism and avoid lookahead
    for idx in range(len(close)):
        if in_position:
            # If currently long, check for exit
            if bool(exit_signal.iloc[idx]):
                in_position = False
                pos.iloc[idx] = 0
            else:
                pos.iloc[idx] = 1
        else:
            # If currently flat, check for entry
            if bool(entry_signal.iloc[idx]):
                in_position = True
                pos.iloc[idx] = 1
            else:
                pos.iloc[idx] = 0

    # Ensure no NaNs in position (use 0 for any remaining NaNs)
    pos = pos.fillna(0).astype(int)

    return {"ohlcv": pos}
