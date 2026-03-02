"""
RSI Mean Reversion Signal Generator

Implements RSI-based mean reversion signals (long-only):
- Compute RSI using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
- Entry when RSI crosses below oversold threshold
- Exit when RSI crosses above overbought threshold

Exports:
- generate_signals(data: dict, params: dict) -> dict

The function expects data["ohlcv"] to be a DataFrame with a 'close' column.
Returns {"ohlcv": position_series} where position_series contains 0 (flat) or 1 (long).

This implementation avoids lookahead by using only past data for RSI computation
and uses a single-pass state machine to produce positions.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    This implementation is causal (no lookahead) and returns a Series aligned
    with the input 'close'. Values before enough data are NaN.

    Args:
        close: Series of close prices.
        period: RSI lookback period (e.g., 14).

    Returns:
        pd.Series: RSI values (0-100) with same index as close.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: exponential moving average with alpha=1/period
    # adjust=False -> recursive calculation that only uses past values
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI calculation
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss == 0 -> RSI = 100 (all gains)
    # If both avg_gain and avg_loss are 0 (flat), set RSI to 50
    mask_loss_zero = (avg_loss == 0) & (avg_gain != 0)
    mask_both_zero = (avg_loss == 0) & (avg_gain == 0)

    rsi = rsi.copy()
    rsi[mask_loss_zero] = 100.0
    rsi[mask_both_zero] = 50.0

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only positions based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" -> DataFrame containing at least 'close' column.
        params: dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        dict with key "ohlcv" -> pd.Series of positions (0 or 1), same index as input.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame value")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    if not (0 <= oversold <= 100 and 0 <= overbought <= 100):
        raise ValueError("oversold and overbought must be between 0 and 100")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions (use previous bar to detect crossings)
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold threshold (from above to below)
    entry_signals = (rsi_prev > oversold) & (rsi <= oversold)

    # Exit: RSI crosses above overbought threshold (from below to above)
    exit_signals = (rsi_prev < overbought) & (rsi >= overbought)

    # Build position series using a single-pass state machine to avoid double entries
    n = len(close)
    positions = np.zeros(n, dtype=int)

    prev_pos = 0  # 0 = flat, 1 = long
    # Iterate by integer index for deterministic behavior
    for i in range(n):
        # If previous position is flat and we have an entry signal -> go long
        if prev_pos == 0 and bool(entry_signals.iloc[i]):
            curr_pos = 1
        # If previous position is long and we have an exit signal -> go flat
        elif prev_pos == 1 and bool(exit_signals.iloc[i]):
            curr_pos = 0
        else:
            # Otherwise, maintain previous position
            curr_pos = prev_pos

        positions[i] = curr_pos
        prev_pos = curr_pos

    position_series = pd.Series(positions, index=close.index, name="position")

    # Ensure output is clean: positions only contain 0 or 1 and no NaNs
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
