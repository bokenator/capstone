import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy (long-only).

    Logic:
    - Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    - Entry: RSI crosses BELOW oversold threshold (prev >= oversold and curr < oversold)
    - Exit: RSI crosses ABOVE overbought threshold (prev <= overbought and curr > overbought)

    Args:
        data: Dict with key 'ohlcv' mapping to a DataFrame containing at least 'close' column.
              (Alternatively, a single DataFrame may be provided under the 'ohlcv' key.)
        params: Dict with keys 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed same as input.
    """
    # Validate and extract DataFrame
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with an 'ohlcv' key mapping to a DataFrame")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame value")
    df = data['ohlcv']
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in df.columns:
        raise KeyError("Input DataFrame must contain 'close' column")

    close = df['close'].astype(float)

    # Parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Basic parameter validation (within allowed ranges)
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')
    if not (0.0 <= oversold <= 50.0):
        raise ValueError('oversold must be between 0 and 50')
    if not (50.0 <= overbought <= 100.0):
        raise ValueError('overbought must be between 50 and 100')

    # Compute price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (EMA with alpha=1/period, recursive)
    # Use min_periods=rsi_period so that initial average is NaN until enough data
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative Strength and RSI
    # Handle division by zero: if avg_loss == 0 -> RSI = 100; if avg_gain == 0 -> RSI = 0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss == 0 and avg_gain == 0 (flat prices), set RSI to 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask(both_zero, 50.0)

    # Where avg_loss == 0 (and avg_gain > 0), RSI = 100
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    # Where avg_gain == 0 (and avg_loss > 0), RSI = 0
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)

    # Determine crossovers (use previous bar to avoid lookahead)
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean Series aligned with close index
    entry_signals = entry_signals.fillna(False).astype(bool)
    exit_signals = exit_signals.fillna(False).astype(bool)

    # Build position series by iterating to avoid double entries/exits
    n = len(close)
    pos = np.zeros(n, dtype=int)
    current = 0
    for i in range(n):
        if entry_signals.iat[i] and current == 0:
            current = 1
        elif exit_signals.iat[i] and current == 1:
            current = 0
        pos[i] = current

    position = pd.Series(pos, index=close.index, name='position')

    # Ensure only 0 or 1 values and no NaNs
    position = position.fillna(0).astype(int)

    return {'ohlcv': position}
