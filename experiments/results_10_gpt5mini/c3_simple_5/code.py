import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Strategy:
    - Compute RSI with Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    - Entry when RSI crosses below oversold threshold
    - Exit when RSI crosses above overbought threshold

    Args:
        data: Dict containing 'ohlcv' DataFrame with a 'close' column.
        params: Dict with keys:
            - 'rsi_period' (int): lookback period for RSI
            - 'oversold' (float): RSI oversold threshold (e.g., 30)
            - 'overbought' (float): RSI overbought threshold (e.g., 70)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 = flat, 1 = long),
        indexed the same as the input close series.
    """
    # Input validation
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame value")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].copy()

    # Ensure numeric dtype
    close = pd.to_numeric(close, errors='coerce')

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Handle NaNs in close in a causal way (forward-fill only). Do NOT backfill to avoid lookahead.
    close = close.fillna(method='ffill')

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    # This implementation is causal and does not use future data
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use ewm with adjust=False for Wilder smoothing
    # Use min_periods=1 so we get early values; NaNs may still appear if close starts with NaN
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RSI safely
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where roll_down is zero (no losses), RSI should be 100; where roll_up is zero, RSI should be 0
    rsi = rsi.where(~(roll_down == 0), 100.0)
    rsi = rsi.where(~(roll_up == 0) | (roll_down == 0), 0.0)

    # Prepare crossing signals (use previous bar and current bar -> causal)
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and curr < oversold
    entries = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit when RSI crosses above overbought: prev <= overbought and curr > overbought
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs in signals with False
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Build position series iteratively to ensure no double entries and causal behavior
    positions = pd.Series(0, index=close.index, dtype='int8')

    pos = 0
    # Iterate using index to preserve alignment and avoid lookahead
    for idx in range(len(close)):
        if entries.iat[idx] and pos == 0:
            pos = 1
        elif exits.iat[idx] and pos == 1:
            pos = 0
        positions.iat[idx] = pos

    # Ensure dtype and no NaNs
    positions = positions.astype(int)

    return {'ohlcv': positions}
