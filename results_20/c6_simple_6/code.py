from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series (float), may contain NaNs for all-NaN input.
    """
    # Safety: ensure float dtype
    close = close.astype(float)

    # Price differences
    delta = close.diff()

    # Gains and losses
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use Wilder's smoothing: ewm with alpha=1/period, adjust=False (recursive)
    # This uses only past data and does not introduce lookahead.
    roll_up = up.ewm(alpha=1.0 / float(period), adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / float(period), adjust=False).mean()

    # Avoid division by zero; rs will be inf where roll_down == 0 and roll_up > 0
    rs = roll_up / roll_down

    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position targets (0 or 1) based on RSI mean reversion.

    Strategy rules:
      - Compute RSI with period params['rsi_period']
      - Go long when RSI crosses below params['oversold']
      - Exit (go flat) when RSI crosses above params['overbought']

    Args:
        data: dict with key 'ohlcv' containing a DataFrame with 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1),
        indexed the same as input close prices.
    """
    # Validate input
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Read params with defaults
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period <= 0:
        rsi_period = 14
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Handle trivial case: empty series
    if len(close) == 0:
        return {'ohlcv': pd.Series(dtype='int64')}

    # Compute RSI. Use past-only smoothing; fill initial NaNs with neutral 50.
    rsi = _compute_rsi(close, rsi_period)
    rsi = rsi.fillna(50.0)

    # Previous RSI (lagged by 1). Fill the first prev value with neutral 50 so
    # that an initial oversold bar can trigger an entry if rsi[0] < oversold.
    prev_rsi = rsi.shift(1).fillna(50.0)

    # Entry: previous >= oversold and current < oversold (crossing below)
    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: previous <= overbought and current > overbought (crossing above)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series via a single pass to avoid double entries/exits.
    pos = 0
    positions = np.zeros(len(rsi), dtype=np.int8)

    # Use iloc to be robust to arbitrary indexes
    for i in range(len(rsi)):
        if pos == 0 and bool(entry_cond.iloc[i]):
            pos = 1
        elif pos == 1 and bool(exit_cond.iloc[i]):
            pos = 0
        positions[i] = pos

    position_series = pd.Series(positions.astype(np.int8), index=close.index, name='position')

    return {'ohlcv': position_series}
