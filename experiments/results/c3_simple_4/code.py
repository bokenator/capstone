"""
RSI Mean Reversion Signal Generator

Exports:
- generate_signals(data, params) -> dict with key 'ohlcv' mapping to a pd.Series of position targets.

Position values: 1 = long, 0 = flat (long-only strategy)

Logic:
- RSI period = 14 (Wilder's smoothing via ewm with adjust=False)
- Entry when RSI crosses below 30: (rsi < 30) & (rsi.shift(1) >= 30)
- Exit when RSI crosses above 70: (rsi > 70) & (rsi.shift(1) <= 70)

The function accepts either:
- data: dict with data['ohlcv'] being a DataFrame containing 'close'
- data: pd.Series or pd.DataFrame of close prices
- params: optional dict (not used here, kept for compatibility)

Returns:
- {'ohlcv': position_series}

This implementation is causal (no lookahead), deterministic, and ensures no NaNs in the output (positions are 0/1).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import vectorbt as vbt


def _extract_close(data: Union[Dict[str, pd.DataFrame], pd.Series, pd.DataFrame]) -> pd.Series:
    """Extract close price series from different possible input formats."""
    # dict with 'ohlcv' DataFrame
    if isinstance(data, dict):
        if "ohlcv" in data:
            df = data["ohlcv"]
            if isinstance(df, pd.DataFrame) and "close" in df.columns:
                close = df["close"].copy()
            else:
                raise ValueError("data['ohlcv'] must be a DataFrame with a 'close' column")
        else:
            # try to find first DataFrame with close
            for v in data.values():
                if isinstance(v, pd.DataFrame) and "close" in v.columns:
                    close = v["close"].copy()
                    break
            else:
                raise ValueError("No 'ohlcv' key found and no DataFrame with 'close' column in data dict")
    elif isinstance(data, pd.DataFrame):
        # if DataFrame, expect a single-column close or a 'close' column
        if "close" in data.columns:
            close = data["close"].copy()
        elif data.shape[1] == 1:
            close = data.iloc[:, 0].copy()
        else:
            raise ValueError("DataFrame input must have a 'close' column or be single-column of close prices")
    elif isinstance(data, pd.Series):
        close = data.copy()
    else:
        raise TypeError("Unsupported data type for price input")

    # Ensure sorted index and monotonic
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()

    # Ensure float dtype
    close = close.astype(float)
    return close


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with adjust=False).

    This implementation is causal (uses past data only).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing via EMA with alpha = 1/period and adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # rsi will be NaN when avg_loss is zero and avg_gain is zero (flat series) -> set to 50
    rsi = rsi.fillna(50.0)
    return rsi


def generate_signals(
    data: Union[Dict[str, pd.DataFrame], pd.Series, pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.Series]:
    """Generate long-only position targets based on RSI mean reversion.

    Args:
        data: Input price data. Accepted formats:
            - dict with key 'ohlcv' containing DataFrame with 'close'
            - pd.Series of close prices
            - pd.DataFrame with 'close' column or single-column close
        params: Optional params (unused but kept for compatibility)

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (1 = long, 0 = flat)
    """
    close = _extract_close(data)

    period = 14
    rsi = _compute_rsi(close, period=period)

    # Signals: cross below 30 -> entry, cross above 70 -> exit
    rsi_prev = rsi.shift(1)

    entry_signal = (rsi < 30) & (rsi_prev >= 30)
    exit_signal = (rsi > 70) & (rsi_prev <= 70)

    # Ensure boolean series align with close index
    entry_signal = entry_signal.reindex(close.index).fillna(False)
    exit_signal = exit_signal.reindex(close.index).fillna(False)

    # Build position series iteratively to avoid double entries and ensure causality
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_position = False

    for t in range(len(close)):
        if not in_position and entry_signal.iloc[t]:
            in_position = True
            pos.iloc[t] = 1
        elif in_position:
            # stay long unless exit signal
            if exit_signal.iloc[t]:
                in_position = False
                pos.iloc[t] = 0
            else:
                pos.iloc[t] = 1
        else:
            pos.iloc[t] = 0

    # No NaNs and integer positions
    pos = pos.astype(int)

    return {"ohlcv": pos}
