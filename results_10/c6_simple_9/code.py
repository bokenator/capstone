from typing import Dict, Any
import pandas as pd
import numpy as np


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: Lookback period for RSI.

    Returns:
        RSI as a pandas Series aligned with `close` index.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm with alpha=1/period, adjust=False for recursive calculation
    # Use min_periods=period so RSI is NaN until we have enough data
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))

    # If both roll_up and roll_down are zero, set RSI to 50 (no movement)
    both_zero = (roll_up == 0) & (roll_down == 0)
    if both_zero.any():
        rsi = rsi.copy()
        rsi[both_zero] = 50.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Strategy:
    - Compute RSI with period `rsi_period`.
    - Go long when RSI crosses below `oversold` (i.e., prev >= oversold and current < oversold).
    - Exit long when RSI crosses above `overbought` (i.e., prev <= overbought and current > overbought).

    Args:
        data: Dict containing 'ohlcv' DataFrame with a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of positions (0 or 1), indexed like the input closes.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].astype(float)

    # Read params with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Handle trivial lengths
    n = len(close)
    if n == 0:
        return {'ohlcv': pd.Series(dtype='int64')}
    if n == 1:
        return {'ohlcv': pd.Series([0], index=close.index, dtype='int64')}

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Previous RSI (shifted by 1). Comparisons with NaN will be False which avoids spurious crosses.
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold threshold
    entry_signals = (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought threshold
    exit_signals = (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (0 or 1) with a single pass to avoid double entries
    pos = np.zeros(n, dtype='int64')

    for i in range(n):
        if i == 0:
            pos[i] = 0
            continue

        if entry_signals.iat[i] and pos[i - 1] == 0:
            pos[i] = 1
        elif exit_signals.iat[i] and pos[i - 1] == 1:
            pos[i] = 0
        else:
            pos[i] = pos[i - 1]

    position_series = pd.Series(pos, index=close.index, name='position')

    # Ensure output contains only 0 or 1 and no NaNs
    position_series = position_series.fillna(0).astype('int64')

    return {'ohlcv': position_series}
