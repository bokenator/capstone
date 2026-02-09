import numpy as np
import pandas as pd
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    This implementation is causal (no lookahead) and works with pandas Series indices.

    Args:
        close: Series of close prices
        period: RSI lookback period

    Returns:
        RSI Series (float), same index as input
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    close = close.astype(float).copy()

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (EWMA with alpha = 1/period, adjust=False)
    # This is causal: value at time t depends only on values <= t
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Handle divide-by-zero cases:
    # - If avg_loss == 0 and avg_gain == 0 -> RSI is set to 50 (no movement)
    # - If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)

    rsi = rsi.copy()
    rsi[mask_both_zero] = 50.0
    rsi[mask_loss_zero] = 100.0
    rsi[mask_gain_zero] = 0.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals (0 or 1) based on RSI mean-reversion.

    Strategy:
    - Compute RSI with period params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key 'ohlcv' mapping to a DataFrame that contains a 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), same index as input
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].copy()

    # Extract params with defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    oversold = float(params.get('oversold', 30.0)) if params is not None else 30.0
    overbought = float(params.get('overbought', 70.0)) if params is not None else 70.0

    if len(close) == 0:
        # Return empty series with same index
        return {'ohlcv': pd.Series(dtype=float, index=close.index)}

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Previous RSI (lagged by 1)
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and curr < oversold
    entry = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit when RSI crosses above overbought: prev <= overbought and curr > overbought
    exit_ = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series and align index
    entry = entry.reindex_like(close).fillna(False).astype(bool)
    exit_ = exit_.reindex_like(close).fillna(False).astype(bool)

    # Build position series (0 or 1) in a causal loop (depends only on past/current signals)
    n = len(close)
    pos_arr = np.zeros(n, dtype=int)
    entry_arr = entry.to_numpy(dtype=bool)
    exit_arr = exit_.to_numpy(dtype=bool)

    for i in range(n):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = pos_arr[i - 1]

        if prev_pos == 0:
            # Only enter if flat and entry signal triggers
            if entry_arr[i]:
                pos_arr[i] = 1
            else:
                pos_arr[i] = 0
        else:
            # Only exit if long and exit signal triggers
            if exit_arr[i]:
                pos_arr[i] = 0
            else:
                pos_arr[i] = 1

    positions = pd.Series(pos_arr, index=close.index, name='position')

    # Final sanity: ensure values are only 0 or 1 and no NaN
    positions = positions.fillna(0).astype(int)

    return {'ohlcv': positions}
