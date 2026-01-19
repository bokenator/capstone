"""
RSI Mean Reversion Strategy Signal Generator

This module exports a single function `generate_signals` which supports multiple calling
conventions to be compatible with the test harness and the backtest runner:

- generate_signals(data_dict: dict, params: dict) -> dict[str, pd.Series]
    Used by the backtest runner. Expects data_dict["ohlcv"]["close"] to exist.
    Returns {"ohlcv": position_series} where position_series values are 1 (long) or 0 (flat).

- generate_signals(df: pd.DataFrame) -> pd.Series
    If a DataFrame (with a 'close' column) is provided and no params, returns the position
    series directly. Used by invariant tests which expect a single-series output.

- generate_signals(close: pd.Series) -> tuple(pd.Series, pd.Series)
    If a plain close price Series is provided and no params, returns a tuple
    (entries, exits) of boolean Series. Used by strategy-specific unit tests.

Strategy:
- RSI period = 14
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long-only, single asset

The implementation uses Wilder's smoothing (EMA with alpha=1/period) for RSI
and avoids any lookahead. Positions are generated iteratively to prevent double
entries (no entry when already long).

Edge cases handled:
- Accepts pd.Series, pd.DataFrame or data dict with 'ohlcv' key
- Ensures returned Series align with input index
- No NaNs in returned position/entry/exit Series (positions filled with 0 where undefined)

"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA) without lookahead.

    Args:
        close: Price series.
        period: RSI period (default 14).

    Returns:
        RSI series aligned with close.index.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's EMA (alpha = 1/period)
    # Use adjust=False to be recursive (no lookahead)
    avg_gain = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is zero (no losses), RSI should be 100; where avg_gain is zero, RSI 0.
    rsi = rsi.fillna(0)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0

    return rsi


def _generate_position_from_signals(
    entries: pd.Series, exits: pd.Series
) -> pd.Series:
    """Create a position series (0/1) from entry/exit boolean signals.

    Ensures no double entries: an entry while already in position is ignored.

    Args:
        entries: Boolean Series indicating desired entry points.
        exits: Boolean Series indicating desired exit points.

    Returns:
        position: Integer Series (0 or 1) aligned with entries.index.
    """
    idx = entries.index
    pos = np.zeros(len(idx), dtype=np.int8)

    in_position = False
    for i in range(len(idx)):
        if entries.iat[i] and not in_position:
            in_position = True
        if exits.iat[i] and in_position:
            in_position = False
        pos[i] = 1 if in_position else 0

    return pd.Series(pos, index=idx, name="position")


def generate_signals(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series],
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, pd.Series], pd.Series, Tuple[pd.Series, pd.Series]]:
    """Generate RSI mean-reversion trading signals.

    Flexible calling conventions:
    - If `data` is a dict and `params` provided: returns a dict with key 'ohlcv'
      mapping to the position series (for backtest runner).
    - If `data` is a DataFrame and `params` is None: returns the position Series.
    - If `data` is a Series (close prices) and `params` is None: returns (entries, exits)
      boolean Series tuple (used by unit tests that expect separate signals).

    Args:
        data: Price input. Can be:
            - dict with key 'ohlcv' -> DataFrame containing 'close' column (backtest)
            - DataFrame with 'close' column
            - Series of close prices
        params: Optional params (unused here, present for compatibility)

    Returns:
        Depending on input shape as described above.
    """
    # Extract close price series depending on input type
    close: pd.Series
    returned_as_backtest = False

    if isinstance(data, dict):
        # Backtest runner convention: data['ohlcv'] is a DataFrame with 'close'
        if "ohlcv" not in data:
            raise ValueError("Input data dict must contain 'ohlcv' key")
        ohlcv = data["ohlcv"]
        if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
            raise ValueError("data['ohlcv'] must be a DataFrame containing 'close' column")
        close = ohlcv["close"].astype(float)
        returned_as_backtest = True
    elif isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame input must contain a 'close' column")
        close = data["close"].astype(float)
    elif isinstance(data, pd.Series):
        close = data.astype(float)
    else:
        raise TypeError("Unsupported data type for generate_signals")

    # Ensure index is monotonic increasing; work with a copy to avoid mutating input
    close = close.copy()

    # Parameters (fixed per prompt)
    rsi_period = 14
    rsi_oversold = 30.0
    rsi_overbought = 70.0

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Determine entry/exits by crossover using only past information
    prev_rsi = rsi.shift(1)

    entries = (prev_rsi >= rsi_oversold) & (rsi < rsi_oversold)
    exits = (prev_rsi <= rsi_overbought) & (rsi > rsi_overbought)

    # Ensure boolean dtype and align index
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Build position series avoiding double entries
    position = _generate_position_from_signals(entries, exits)

    # Fill any potential NaNs (there shouldn't be) with 0 and cast to int
    position = position.fillna(0).astype(int)

    # If called in backtest mode, return dict
    if returned_as_backtest:
        return {"ohlcv": position}

    # If input was a plain Series and no params supplied, many unit tests expect
    # a tuple of (entries, exits). If input was a DataFrame, unit tests expect a
    # single position Series. We distinguish by the input type above.
    if isinstance(data, pd.Series):
        return entries, exits

    # Default: DataFrame input, return position Series
    return position
