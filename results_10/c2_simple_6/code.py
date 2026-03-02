"""
RSI Mean Reversion Signal Generator

Implements:
- RSI (14)
- Go long when RSI crosses below 30
- Exit (go flat) when RSI crosses above 70
- Long-only, single asset

Exports:
- generate_signals(data, params) -> dict[str, pd.Series]

Notes:
- Imports vectorbt inside the function to avoid import-time errors if vectorbt is
  not installed in the environment importing this module. The backtest runner
  checks for vectorbt availability before executing the strategy.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position targets based on RSI mean-reversion strategy.

    Args:
        data: Dict containing at least the key "ohlcv" with a DataFrame that has
            a "close" column (pd.Series).
        params: Dictionary of parameters (not used here, kept for interface
            compatibility).

    Returns:
        A dictionary with key "ohlcv" mapping to a pd.Series of position targets
        (0 = flat, 1 = long) indexed the same as the input close prices.

    Strategy details:
    - RSI period = 14
    - Entry (go long) when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit (go flat) when RSI crosses above 70 (prev <= 70 and curr > 70)

    Edge cases handled:
    - Warmup NaN values in RSI: no entries/exits until RSI becomes valid
    - If required keys/columns are missing, raises informative errors
    """
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].copy()

    # Ensure numeric close series
    close = pd.to_numeric(close, errors="coerce")

    # Import vectorbt locally to avoid import-time failures in environments
    # where vectorbt isn't installed. The backtest runner verifies vectorbt
    # availability before calling this function.
    try:
        import vectorbt as vbt
    except Exception as e:  # pragma: no cover - defensive
        raise ImportError("vectorbt is required by generate_signals but is not installed") from e

    # RSI period fixed at 14 according to the specification
    rsi_period = 14

    # Compute RSI with vectorbt's RSI indicator
    # vbt.RSI.run returns an object with attribute `.rsi` (a Series)
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    # rsi_res.rsi is either a Series (single column) or DataFrame for multiple columns
    rsi = rsi_res.rsi

    # Ensure we have a Series (single asset). If a DataFrame is returned with a
    # single column, convert to Series.
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            # Multi-column RSI not supported by this simple single-asset strategy
            raise ValueError("RSI returned multiple columns; expected a single asset")

    # Align indices
    rsi = rsi.reindex(close.index)

    # Previous RSI for crossing detection
    prev_rsi = rsi.shift(1)

    # Define entry/exit conditions, require both prev and curr RSI to be non-null
    entry_cond = prev_rsi.notna() & rsi.notna() & (prev_rsi >= 30) & (rsi < 30)
    exit_cond = prev_rsi.notna() & rsi.notna() & (prev_rsi <= 70) & (rsi > 70)

    # Build position series by iterating once through the bars to ensure
    # correct stateful behavior (long-only)
    pos_values: list[int] = []
    in_position = 0

    # Iterate in chronological order (index is assumed sorted)
    for idx in rsi.index:
        if in_position == 0 and bool(entry_cond.loc[idx]):
            in_position = 1
        elif in_position == 1 and bool(exit_cond.loc[idx]):
            in_position = 0
        pos_values.append(in_position)

    position = pd.Series(pos_values, index=rsi.index, name="position").astype(int)

    return {"ohlcv": position}
