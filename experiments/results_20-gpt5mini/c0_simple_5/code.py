"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean reversion strategy:
- RSI period = 14 (configurable via params)
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)
- Long-only, single-asset

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

The function expects `data` to contain an 'ohlcv' key with a pandas DataFrame
that includes a 'close' column. It returns a dict with key 'ohlcv' whose value
is a pandas Series of positions (1 = long, 0 = flat) aligned with the close
series index.

This module uses pandas and numpy only to avoid hard dependency issues at import
time. The backtest runner will use vectorbt to execute the backtest.
"""

from typing import Any
import pandas as pd
import numpy as np


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        close: Price series (pd.Series).
        period: RSI lookback period.

    Returns:
        RSI series (pd.Series) aligned with `close`. The first `period` bars are
        set to NaN to represent the warmup period.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    if period <= 0:
        raise ValueError("period must be a positive integer")

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0).fillna(0.0)
    loss = -delta.clip(upper=0).fillna(0.0)

    # Wilder's smoothing via exponential moving average with alpha = 1/period
    # This approximates the original Wilder RSI smoothing approach.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Enforce warmup NaNs for the initial period
    rsi.iloc[:period] = np.nan

    return rsi


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - Compute RSI with period (default 14)
    - Entry: RSI crosses below oversold level (default 30)
    - Exit: RSI crosses above overbought level (default 70)

    Args:
        data: Dictionary expected to contain key 'ohlcv' with a pd.DataFrame
            that includes a 'close' column.
        params: Dictionary of parameters. Recognized keys:
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'rsi_oversold' (float): Oversold threshold. Default 30.
            - 'rsi_overbought' (float): Overbought threshold. Default 70.

    Returns:
        A dictionary with key 'ohlcv' mapping to a pd.Series of positions
        (1 = long, 0 = flat) aligned with the input close prices.
    """
    # Basic validation
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing an 'ohlcv' DataFrame")

    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame value")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with sensible defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('rsi_oversold', 30.0))
    overbought = float(params.get('rsi_overbought', 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")
    if not (0 <= oversold < overbought <= 100):
        raise ValueError("rsi_oversold and rsi_overbought must satisfy 0 <= oversold < overbought <= 100")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crossings: entry when RSI crosses below oversold, exit when RSI crosses above overbought
    prev_rsi = rsi.shift(1)

    # Entry: previous >= oversold and current < oversold
    entries = (prev_rsi >= oversold) & (rsi < oversold) & rsi.notna()

    # Exit: previous <= overbought and current > overbought
    exits = (prev_rsi <= overbought) & (rsi > overbought) & rsi.notna()

    # Build position series: 1 while in a long trade, 0 otherwise
    position = pd.Series(0, index=close.index, dtype='int8')

    in_trade = False
    # Iterate in order to properly handle stateful entries/exits
    for idx in close.index:
        if (not in_trade) and entries.loc[idx]:
            in_trade = True
        elif in_trade and exits.loc[idx]:
            in_trade = False
        position.loc[idx] = 1 if in_trade else 0

    # Ensure dtype int and alignment with close index
    position = position.astype(int)

    return {"ohlcv": position}
