"""
RSI Mean Reversion Signal Generator

Implements generate_signals(data: dict, params: dict) -> dict
- data["ohlcv"]: pd.DataFrame with 'close' column
- params: contains 'rsi_period', 'oversold', 'overbought'

Returns {"ohlcv": position_series} where position_series values are 0 (flat) or 1 (long)

Notes:
- RSI computed using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
- Entry: RSI crosses below oversold threshold (prev >= oversold and curr < oversold)
- Exit: RSI crosses above overbought threshold (prev <= overbought and curr > overbought)
- Deterministic, causal (no lookahead), handles NaNs/warmup periods
"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    This implementation is causal (uses only past data) and avoids lookahead bias.

    Parameters
    ----------
    close : pd.Series
        Close prices (indexed)
    period : int
        RSI period (e.g., 14)

    Returns
    -------
    pd.Series
        RSI values (float), may contain NaN for very early points
    """
    if period <= 0:
        raise ValueError("rsi period must be > 0")

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: exponential moving average with alpha = 1/period
    # adjust=False ensures the calculation is recursive and causal
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Prepare RSI series
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss == 0
    # If avg_loss == 0 and avg_gain == 0 -> RSI is neutral (50)
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    mask_loss_zero = (avg_loss == 0) & avg_gain.notna()
    if mask_loss_zero.any():
        rsi = rsi.copy()
        both_zero = mask_loss_zero & (avg_gain == 0)
        only_gain = mask_loss_zero & (avg_gain > 0)
        rsi.loc[both_zero] = 50.0
        rsi.loc[only_gain] = 100.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Parameters
    ----------
    data : dict
        Must contain key 'ohlcv' mapping to a DataFrame with a 'close' column.
    params : dict
        Parameters for the strategy. Expected keys:
        - 'rsi_period' (int): RSI lookback period
        - 'oversold' (float): oversold threshold (e.g., 30.0)
        - 'overbought' (float): overbought threshold (e.g., 70.0)

    Returns
    -------
    dict
        {'ohlcv': position_series} where position_series is a pd.Series of 0/1
    """
    # Validate input
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].copy()

    # Parameters with sensible defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    oversold = float(params.get('oversold', 30.0)) if params is not None else 30.0
    overbought = float(params.get('overbought', 70.0)) if params is not None else 70.0

    if rsi_period <= 0:
        raise ValueError('rsi_period must be a positive integer')

    # Compute RSI (causal)
    rsi = _compute_rsi_wilder(close, rsi_period)

    # Prepare position series (0 = flat, 1 = long)
    n = len(close)
    positions = pd.Series(0, index=close.index, dtype='int8')

    # Stateful simulation to enforce single-entry behavior (no double entries)
    current_pos = 0

    # Iterate over bars
    # Use iloc for speed and deterministic indexing
    for i in range(n):
        curr_rsi = rsi.iloc[i]
        prev_rsi = rsi.iloc[i - 1] if i > 0 else np.nan

        # Only act on valid RSI values (skip if NaN)
        if np.isnan(curr_rsi) or np.isnan(prev_rsi):
            # Keep current position (no change)
            positions.iloc[i] = current_pos
            continue

        if current_pos == 0:
            # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
            if (prev_rsi >= oversold) and (curr_rsi < oversold):
                current_pos = 1
        else:
            # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
            if (prev_rsi <= overbought) and (curr_rsi > overbought):
                current_pos = 0

        positions.iloc[i] = current_pos

    # Ensure positions contain only 0 or 1 and align with input index/length
    positions = positions.astype('int8')

    return {'ohlcv': positions}
