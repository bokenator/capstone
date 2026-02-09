"""
RSI Mean Reversion Signal Generator

Implements an RSI(14) mean-reversion strategy:
- Go long when RSI crosses below the oversold threshold (default 30)
- Exit when RSI crosses above the overbought threshold (default 70)
- Long-only, single-asset

Function exported: generate_signals(data: dict, params: dict) -> dict
- data["ohlcv"] is expected to be a pandas DataFrame with a 'close' column
- params: {"rsi_period": int, "oversold": float, "overbought": float}
- Returns {"ohlcv": position_series} where position_series contains 0 or 1

The implementation is careful to avoid lookahead bias: signals at time t only
use data at times <= t. It also ensures no NaNs in the returned position series.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    Parameters
    ----------
    close : pd.Series
        Close price series
    period : int
        RSI period (typical 14)

    Returns
    -------
    pd.Series
        RSI values (may contain NaN at the very beginning if input is too short)
    """
    close = close.astype(float).copy()
    # Price differences
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm with alpha=1/period; adjust=False ensures
    # recursive calculation equivalent to Wilder's moving average.
    # This uses only past values (no lookahead).
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position series based on RSI mean reversion.

    Parameters
    ----------
    data : dict
        Dictionary containing 'ohlcv' DataFrame with a 'close' column.
    params : dict
        Parameters with keys:
        - 'rsi_period' : int
        - 'oversold' : float
        - 'overbought' : float

    Returns
    -------
    dict
        {'ohlcv': position_series} where position_series is a pd.Series of 0/1
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with an 'ohlcv' DataFrame")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Read parameters with safe defaults
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        raise ValueError("params must contain numeric 'rsi_period', 'oversold', 'overbought'")

    if rsi_period <= 0:
        raise ValueError("rsi_period must be > 0")

    # Compute RSI (uses only past data via ewm with adjust=False)
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crossings using only current and previous RSI values (no future data)
    prev_rsi = rsi.shift(1)

    # Cross below oversold: prev >= oversold and current < oversold
    cross_below_oversold = (rsi < oversold) & (prev_rsi >= oversold)

    # Cross above overbought: prev <= overbought and current > overbought
    cross_above_overbought = (rsi > overbought) & (prev_rsi <= overbought)

    # Initialize positions array (0 = flat, 1 = long)
    n = len(close)
    positions = np.zeros(n, dtype=np.int8)

    # Iterate forward in time to build position series without lookahead.
    # We ensure we only enter when flat and only exit when long.
    pos = 0
    # Use iloc for positional access and preserve index mapping later
    for i in range(n):
        # Only change position based on information up to and including i
        if pos == 0 and cross_below_oversold.iloc[i]:
            pos = 1
        elif pos == 1 and cross_above_overbought.iloc[i]:
            pos = 0
        positions[i] = pos

    position_series = pd.Series(positions, index=close.index, name="position")

    # Ensure only 0/1 values and no NaNs
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
