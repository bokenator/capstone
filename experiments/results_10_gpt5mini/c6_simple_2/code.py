# Generated RSI mean reversion signal generator
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI as a pandas Series aligned with `close`.
    """
    if period <= 0:
        raise ValueError("rsi period must be > 0")

    close = close.astype(float)
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing -> use ewm with alpha=1/period (adjust=False)
    # This is causal and does not use future data.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss == 0 -> RSI is 100; where avg_gain == 0 -> RSI is 0
    rsi = rsi.fillna(0)
    rsi = rsi.where(avg_loss != 0, 100)

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position targets (0 or 1) based on RSI mean reversion.

    Strategy rules:
    - Compute RSI with period params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict containing 'ohlcv' DataFrame with a 'close' column.
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' mapping to a pandas Series of positions (0 or 1),
        index-aligned with input ohlcv.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise KeyError("data must contain key 'ohlcv' with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Read parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if not (0 <= oversold < overbought <= 100):
        raise ValueError("oversold and overbought must satisfy 0 <= oversold < overbought <= 100")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Shifted RSI for detecting crosses (previous value)
    rsi_prev = rsi.shift(1)

    # Detect crossing events
    enter_cross = (rsi_prev >= oversold) & (rsi < oversold)
    exit_cross = (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (0/1) iteratively to avoid lookahead
    n = len(close)
    positions = pd.Series(0, index=close.index, dtype=int)

    prev_pos = 0
    for i, idx in enumerate(close.index):
        # By default carry forward previous position
        pos = prev_pos

        # Only consider crosses where previous RSI exists
        if i == 0:
            # At first bar, we have no prior RSI to determine a cross; stay flat
            pos = 0
        else:
            if prev_pos == 0:
                # If flat, enter on oversold cross
                if bool(enter_cross.iloc[i]):
                    pos = 1
                else:
                    pos = 0
            else:
                # If long, exit on overbought cross
                if bool(exit_cross.iloc[i]):
                    pos = 0
                else:
                    pos = 1

        positions.iloc[i] = pos
        prev_pos = pos

    # Ensure positions are ints and contain no NaN
    positions = positions.fillna(0).astype(int)

    return {'ohlcv': positions}
