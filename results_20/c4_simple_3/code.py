import pandas as pd
import numpy as np
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        close: Price series (pd.Series) of closes.
        period: Lookback period for RSI (int).

    Returns:
        pd.Series of RSI values aligned with `close` index. Initial periods will be NaN.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (exponential moving average with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is 0 (no losses), RSI should be 100. Where avg_gain is 0 (no gains), RSI should be 0.
    rsi = rsi.where(~(avg_loss == 0), 100.0)
    rsi = rsi.where(~(avg_gain == 0), 0.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Default params from PARAM_SCHEMA
    defaults = {
        'rsi_period': 14,
        'oversold': 30.0,
        'overbought': 70.0,
    }

    # Extract and validate params
    rsi_period = params.get('rsi_period', defaults['rsi_period'])
    oversold = params.get('oversold', defaults['oversold'])
    overbought = params.get('overbought', defaults['overbought'])

    try:
        rsi_period = int(rsi_period)
    except Exception:
        raise TypeError("rsi_period must be an integer")

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100 (inclusive)")

    try:
        oversold = float(oversold)
        overbought = float(overbought)
    except Exception:
        raise TypeError("oversold and overbought must be numeric (float)")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")

    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    if not (oversold < overbought):
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (previous >= oversold and current < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (previous <= overbought and current > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Make boolean series have explicit False where NaN
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series: we're long when the number of entries so far > number of exits so far
    entry_counts = entries.astype(int).cumsum()
    exit_counts = exits.astype(int).cumsum()

    position = (entry_counts > exit_counts).astype(int)

    # Ensure alignment and dtype
    position = pd.Series(position, index=close.index, name='position', dtype='int64')

    return {'ohlcv': position}
