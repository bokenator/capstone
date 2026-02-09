from typing import Dict, Any
import pandas as pd
import numpy as np


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing approximation via EWM.

    This implementation uses only past data (no lookahead).
    Returns a Series aligned with `close`.
    """
    close = close.astype(float)
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use exponential weighted moving average with adjust=False to mimic Wilder's smoothing
    # Setting alpha = 1/period approximates Wilder's smoothing factor.
    alpha = 1.0 / float(period) if period > 0 else 1.0
    try:
        avg_gain = up.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = down.ewm(alpha=alpha, adjust=False).mean()
    except TypeError:
        # Fallback if pandas version does not accept alpha directly: use span approximation
        # This fallback is conservative and still uses only past data.
        span = max(1, int(round((2.0 - alpha) / alpha)))
        avg_gain = up.ewm(span=span, adjust=False).mean()
        avg_loss = down.ewm(span=span, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.astype(float)
    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position series based on RSI mean reversion.

    Args:
        data: dict with key 'ohlcv' -> DataFrame containing 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' -> pd.Series of positions (0 for flat, 1 for long)
    """
    # Validate input
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].copy()
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Extract parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Edge cases
    n = len(close)
    if n == 0:
        return {'ohlcv': pd.Series(dtype=int)}

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Determine entry and exit points using crossing logic (no lookahead)
    # Entry: RSI crosses below oversold: rsi[t] < oversold and rsi[t-1] >= oversold
    # Exit: RSI crosses above overbought: rsi[t] > overbought and rsi[t-1] <= overbought
    prev_rsi = rsi.shift(1)

    entry_signals = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signals = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series iteratively to avoid double entries and ensure simple behavior
    positions = np.zeros(n, dtype=int)
    position = 0
    for i in range(n):
        if position == 0:
            if bool(entry_signals.iloc[i]):
                position = 1
        else:  # position == 1
            if bool(exit_signals.iloc[i]):
                position = 0
        positions[i] = position

    position_series = pd.Series(positions, index=close.index, dtype=int)

    return {'ohlcv': position_series}
