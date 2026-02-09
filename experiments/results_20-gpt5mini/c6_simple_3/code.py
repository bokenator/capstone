import numpy as np
import pandas as pd
from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean-reversion.

    Strategy:
    - Compute RSI (Wilder's smoothing via EMA with alpha=1/period)
    - Enter long (1) when RSI crosses below `oversold` (default 30)
    - Exit to flat (0) when RSI crosses above `overbought` (default 70)

    Args:
        data: Dictionary containing 'ohlcv' DataFrame with a 'close' column.
        params: Dictionary with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of positions (0 or 1),
        indexed the same as input 'close'.
    """
    # Validate inputs
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Handle trivial case
    if len(close) == 0:
        return {'ohlcv': pd.Series(dtype=float)}

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = -delta.clip(upper=0).fillna(0.0)

    # Use ewm with adjust=False for a causal (no-lookahead) calculation
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss == 0 and avg_gain == 0 -> set RSI to 50 (neutral)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # Where avg_loss == 0 (and avg_gain > 0) -> RSI = 100
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, rsi))
    # Where avg_gain == 0 (and avg_loss > 0) -> RSI = 0
    rsi = rsi.where(avg_gain != 0, np.where(avg_loss > 0, 0.0, rsi))

    rsi = pd.Series(rsi, index=close.index)

    # Determine cross signals (treat initial valid RSI below/above thresholds as entry/exit)
    rsi_prev = rsi.shift(1)

    enter_signal = (rsi < oversold) & (
        (rsi_prev >= oversold) | (rsi_prev.isna())
    )
    exit_signal = (rsi > overbought) & (
        (rsi_prev <= overbought) | (rsi_prev.isna())
    )

    # Build position series ensuring no double entries/exits and long-only behavior
    positions = np.zeros(len(close), dtype=int)
    pos = 0
    for i in range(len(close)):
        if pos == 0 and enter_signal.iat[i]:
            pos = 1
        elif pos == 1 and exit_signal.iat[i]:
            pos = 0
        positions[i] = pos

    position_series = pd.Series(positions, index=close.index, name='position').astype(int)

    return {'ohlcv': position_series}
