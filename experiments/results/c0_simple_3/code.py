"""
RSI Mean Reversion Signal Generator

Implements a long-only RSI mean reversion strategy:
- RSI period = 14
- Enter long when RSI crosses below 30
- Exit long when RSI crosses above 70

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

The function returns a dict with key "ohlcv" mapping to a pd.Series of position targets
(+1 for long, 0 for flat). The series is aligned with data['ohlcv'].index.

Handles NaNs and warmup periods gracefully.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI using EWM smoothing.

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close.index. Fills initial NaNs with 50.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing via EWM with alpha=1/period
    # Set min_periods=period to avoid early biased values
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Handle divide-by-zero cases: if avg_loss == 0 -> RSI = 100, if both zero -> RSI = 50
    rsi = rsi.copy()
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0

    # Fill initial periods (where min_periods not met) with neutral 50 to avoid false signals
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dict containing at least the key 'ohlcv' -> DataFrame with a 'close' column/Series.
        params: Dict of parameters. Optional keys:
            - 'rsi_period' (int): RSI period. Default 14.
            - 'rsi_oversold' (float): Oversold threshold. Default 30.
            - 'rsi_overbought' (float): Overbought threshold. Default 70.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (+1 long, 0 flat).

    Raises:
        ValueError: if required data is missing or malformed.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    # Extract close series
    # Support both DataFrame column (ohlcv['close']) or Series (ohlcv if already a Series)
    if 'close' in ohlcv.columns:
        close = ohlcv['close'].astype(float)
    elif isinstance(ohlcv, pd.Series):
        close = ohlcv.astype(float)
    else:
        raise ValueError("ohlcv must contain a 'close' column or be a pandas Series")

    # Read params with defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    oversold = float(params.get('rsi_oversold', 30.0)) if params is not None else 30.0
    overbought = float(params.get('rsi_overbought', 70.0)) if params is not None else 70.0

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Determine cross events
    prev_rsi = rsi.shift(1).fillna(method='bfill')  # backfill the first value to avoid NaN comparisons

    # Entry: RSI crosses below oversold (was >= oversold, now < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (was <= overbought, now > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series by simulating the position state machine
    pos = pd.Series(0, index=close.index, dtype='int8')
    in_long = False

    # Use .iloc indexing for speed
    entries_i = entries.astype(bool).to_numpy()
    exits_i = exits.astype(bool).to_numpy()

    for i in range(len(pos)):
        if entries_i[i] and not in_long:
            in_long = True
        elif exits_i[i] and in_long:
            in_long = False
        pos.iloc[i] = 1 if in_long else 0

    # Ensure alignment and no NaNs
    pos = pos.reindex(close.index).fillna(0).astype(int)

    return {'ohlcv': pos}
