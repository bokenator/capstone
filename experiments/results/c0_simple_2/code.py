import pandas as pd
import numpy as np
from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy rules:
    - RSI period = 14 (can be overridden by params['rsi_period'])
    - Go long when RSI crosses below 30 (from >=30 to <30)
    - Exit long when RSI crosses above 70 (from <=70 to >70)
    - Long-only, single asset

    Args:
        data: Dictionary containing price data. Must contain key 'ohlcv' with a
              DataFrame that has a 'close' column (or be a Series itself).
        params: Parameters dictionary. Supported keys:
            - 'rsi_period' (int, optional): RSI lookback period. Default 14.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (1 for long,
        0 for flat) aligned with the input close series index.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise KeyError("data must contain key 'ohlcv' with price data")

    ohlcv = data['ohlcv']

    # Extract close series
    if isinstance(ohlcv, pd.DataFrame):
        if 'close' not in ohlcv.columns:
            raise KeyError("ohlcv DataFrame must contain a 'close' column")
        close = ohlcv['close'].copy()
    elif isinstance(ohlcv, pd.Series):
        close = ohlcv.copy()
    else:
        raise ValueError("data['ohlcv'] must be a pandas DataFrame or Series")

    # Ensure proper dtype and index
    close = close.astype(float)

    # Parameters
    period = int(params.get('rsi_period', 14)) if isinstance(params, dict) else 14
    if period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Handle edge cases: if series too short, return flat positions
    if close.dropna().shape[0] < period + 1:
        # Not enough data to compute RSI reliably
        positions = pd.Series(0, index=close.index, dtype=int)
        return {'ohlcv': positions}

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (alpha = 1/period) and adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss == 0 -> RSI = 100 (strong uptrend), where avg_gain == 0 -> RSI = 0
    rsi = rsi.fillna(0)
    rsi[avg_loss == 0] = 100

    # Detect crossovers
    rsi_prev = rsi.shift(1)
    entry_signals = (rsi_prev >= 30) & (rsi < 30)
    exit_signals = (rsi_prev <= 70) & (rsi > 70)

    # Ensure boolean Series align and have no NaNs (treat NaN as False)
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series iteratively to enforce single-entry while long
    positions = pd.Series(0, index=close.index, dtype=int)
    long = 0
    for ts in close.index:
        if long == 0 and entry_signals.loc[ts]:
            long = 1
        elif long == 1 and exit_signals.loc[ts]:
            long = 0
        positions.loc[ts] = long

    return {'ohlcv': positions}
