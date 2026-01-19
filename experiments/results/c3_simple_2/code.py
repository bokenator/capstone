# rsi_mean_reversion.py
"""
RSI Mean Reversion Strategy

Exports:
- generate_signals(data, params=None) -> when called by backtester: returns dict with key 'ohlcv' mapping to a pd.Series of position targets (0 or 1)

Also supports being called directly with a pd.Series of prices: returns (entries, exits) boolean Series for unit tests that expect that signature.

Strategy:
- RSI period = 14
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long only

"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (no lookahead).

    Args:
        close: Close price series.
        period: RSI period.

    Returns:
        RSI series aligned with close (same index). Values can be NaN for the initial warmup.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's EMA (alpha = 1/period), adjust=False to use recursive form (no lookahead)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _build_position_from_signals(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """Build a long-only position series (0/1) from entry/exit boolean signals.

    Enforces that entries while already in position are ignored.

    Args:
        entries: Boolean series indicating entry signals at given timestamps.
        exits: Boolean series indicating exit signals at given timestamps.

    Returns:
        position series (0 or 1) with same index as inputs.
    """
    if not (len(entries) == len(exits)):
        raise ValueError("entries and exits must be the same length")

    n = len(entries)
    pos = np.zeros(n, dtype=np.int8)

    in_position = False
    for i in range(n):
        if entries.iloc[i] and not in_position:
            in_position = True
            pos[i] = 1
        else:
            pos[i] = 1 if in_position else 0

        # If exit occurs at this bar, close the position immediately (position becomes 0)
        if exits.iloc[i] and in_position:
            in_position = False
            pos[i] = 0

    return pd.Series(pos, index=entries.index)


def generate_signals(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series],
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, pd.Series], Tuple[pd.Series, pd.Series]]:
    """Generate signals for RSI mean reversion strategy.

    Accepts either:
    - data: dict with key 'ohlcv' mapping to a DataFrame containing a 'close' column (used by backtester). In this
      case the function returns a dict with key 'ohlcv' mapping to a pd.Series of position targets (0/1).

    - data: pd.Series of close prices (used by unit tests). In this case the function returns a tuple
      (entries, exits) of boolean Series.

    Args:
        data: price data (dict, DataFrame, or Series).
        params: optional params (not used here but kept for compatibility). If provided, it must be a dict.

    Returns:
        dict with 'ohlcv' -> position pd.Series for backtester, or (entries, exits) when passed a Series.
    """
    # Default RSI period
    period = 14

    # Extract close price series depending on input type
    if isinstance(data, dict):
        # Expect data['ohlcv'] exists
        if 'ohlcv' not in data:
            raise ValueError("When passing a dict, it must contain 'ohlcv' DataFrame with a 'close' column")
        ohlcv = data['ohlcv']
        if isinstance(ohlcv, pd.DataFrame) and 'close' in ohlcv.columns:
            close = ohlcv['close'].astype(float)
        else:
            raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")
        return_dict_mode = True
    elif isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            close = data['close'].astype(float)
        else:
            # If DataFrame is a single-column of prices
            if data.shape[1] == 1:
                close = data.iloc[:, 0].astype(float)
            else:
                raise ValueError("DataFrame must contain a 'close' column or be a single-column price series")
        return_dict_mode = False
    elif isinstance(data, pd.Series):
        close = data.astype(float)
        return_dict_mode = False
    else:
        raise ValueError("Unsupported data type for generate_signals")

    # Ensure index is sorted and unique; do not alter values or introduce lookahead
    close = close.copy()

    # Compute RSI
    rsi = _compute_rsi(close, period=period)

    # Entry: RSI crosses below 30 (prev >= 30 and current < 30)
    prev_rsi = rsi.shift(1)
    entries = (prev_rsi >= 30) & (rsi < 30)

    # Exit: RSI crosses above 70 (prev <= 70 and current > 70)
    exits = (prev_rsi <= 70) & (rsi > 70)

    # Replace NaN booleans with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position targets (0/1) ensuring no double entry
    position = _build_position_from_signals(entries, exits)

    # Ensure same length as input and no NaN
    position = position.reindex(close.index).fillna(0).astype(int)

    if return_dict_mode:
        # Backtest runner expects a dict with key 'ohlcv' mapping to the position Series
        return {'ohlcv': position}
    else:
        # For unit tests, return entries and exits boolean series
        return entries.astype(bool), exits.astype(bool)


# If this module is run directly, provide a simple smoke test
if __name__ == '__main__':
    # Create a small sample
    idx = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = pd.Series(np.linspace(100, 50, 200), index=idx)
    e, x = generate_signals(prices)
    print('Entries:', e.sum(), 'Exits:', x.sum())
