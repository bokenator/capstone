import pandas as pd
import numpy as np
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (no lookahead).

    Args:
        close: Series of close prices.
        period: RSI period (e.g., 14).

    Returns:
        RSI series aligned with close.index. Values before the first full period are NaN.
    """
    close = close.astype(float)
    n = len(close)
    rsi = pd.Series(np.nan, index=close.index, dtype=float)

    if period < 1:
        raise ValueError("period must be >= 1")

    # Need at least period+1 bars to compute the first RSI value (because diff introduces one NA)
    if n < period + 1:
        return rsi

    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = -delta.clip(upper=0).fillna(0.0)

    g = gain.values
    l = loss.values

    avg_g = np.full(n, np.nan, dtype=float)
    avg_l = np.full(n, np.nan, dtype=float)

    # Initial average gain/loss: simple mean of first 'period' deltas (indexes 1..period)
    # Use nanmean to be robust to any NaNs in the slice
    avg_g[period] = np.nanmean(g[1 : period + 1])
    avg_l[period] = np.nanmean(l[1 : period + 1])

    # Wilder's smoothing for subsequent values (uses only past data)
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + g[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + l[i]) / period

    # Compute RSI from average gains/losses
    for i in range(n):
        if np.isnan(avg_g[i]) or np.isnan(avg_l[i]):
            rsi.iloc[i] = np.nan
            continue
        if avg_l[i] == 0.0:
            # No losses -> RSI = 100 if there were gains, else 50
            rsi.iloc[i] = 100.0 if avg_g[i] > 0 else 50.0
        else:
            rs = avg_g[i] / avg_l[i]
            rsi.iloc[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position signals based on RSI mean-reversion.

    Strategy logic:
    - Compute RSI with period=params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key 'ohlcv' containing a DataFrame with a 'close' column.
        params: dict containing 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), same index as input.
    """
    # Validate input
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict with key 'ohlcv' containing a DataFrame with a 'close' column")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame) or 'close' not in ohlcv:
        raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Params with sensible defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    if not (0.0 <= oversold < overbought <= 100.0):
        raise ValueError("oversold and overbought must satisfy 0 <= oversold < overbought <= 100")

    # Compute RSI (no lookahead)
    rsi = _compute_rsi(close, rsi_period)

    # Simulate positions sequentially to avoid lookahead and double entries
    positions = pd.Series(0, index=close.index, dtype=int)
    in_position = 0
    prev_rsi = np.nan

    # Iterate by position to ensure determinism and no lookahead
    for i in range(len(close)):
        cur_rsi = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else np.nan

        if in_position == 0:
            # Entry when RSI crosses below oversold: previous > oversold and current <= oversold
            if (not np.isnan(prev_rsi)) and (not np.isnan(cur_rsi)) and (prev_rsi > oversold) and (cur_rsi <= oversold):
                in_position = 1
        else:
            # Exit when RSI crosses above overbought: previous < overbought and current >= overbought
            if (not np.isnan(prev_rsi)) and (not np.isnan(cur_rsi)) and (prev_rsi < overbought) and (cur_rsi >= overbought):
                in_position = 0

        positions.iloc[i] = int(in_position)
        prev_rsi = cur_rsi

    # Ensure positions contain only 0 or 1 and have same length as input
    positions = positions.fillna(0).astype(int)

    return {'ohlcv': positions}
