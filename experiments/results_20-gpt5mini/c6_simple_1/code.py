"""
RSI Mean Reversion Signal Generator

Implements RSI (Wilder) and generates long-only position signals:
- Entry: RSI crosses below oversold threshold
- Exit: RSI crosses above overbought threshold

Returns a dictionary with key 'ohlcv' mapping to a pd.Series of 0/1 positions
aligned to the input 'close' series index.

The implementation is stateful (prevents double entries) and uses only past and
current data points (no lookahead). It handles NaNs and warmup periods safely.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (the classic RSI implementation).

    Args:
        close: Series of close prices.
        period: RSI lookback period (e.g., 14).

    Returns:
        pd.Series of RSI values aligned with `close.index`. Values are NaN for
        indices before the first full period can be computed.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")

    close = close.astype(float).copy()
    n = len(close)

    # Short circuit for very short series
    if n == 0:
        return pd.Series(dtype=float, index=close.index)

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Prepare output
    rsi = pd.Series(np.nan, index=close.index)

    # Need at least `period + 1` prices to have `period` deltas (delta[0] is NaN)
    if n <= period:
        return rsi

    gains_arr = gains.to_numpy()
    losses_arr = losses.to_numpy()

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # Initial average gain/loss: mean of the first `period` deltas (indices 1..period)
    # Note: delta[0] is NaN, so we take gains[1:period+1]
    initial_gain = np.nanmean(gains_arr[1 : period + 1])
    initial_loss = np.nanmean(losses_arr[1 : period + 1])

    avg_gain[period] = initial_gain
    avg_loss[period] = initial_loss

    # Wilder's smoothing for subsequent values
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains_arr[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses_arr[i]) / period

    # Compute RSI from smoothed averages
    for i in range(period, n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            rsi.iloc[i] = np.nan
            continue
        if al == 0:
            # No losses -> RSI is 100
            rsi_val = 100.0
        else:
            rs = ag / al
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
        rsi.iloc[i] = rsi_val

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data: Dictionary containing at least key "ohlcv" -> DataFrame with a
              'close' column.
        params: Dictionary with keys:
            - 'rsi_period' (int): RSI period (e.g., 14)
            - 'oversold' (float): Oversold threshold (e.g., 30.0)
            - 'overbought' (float): Overbought threshold (e.g., 70.0)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), with
        the same index as the input 'close' series.
    """
    # Basic input validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].copy().astype(float)

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be > 0")
    if not (0 <= oversold < overbought <= 100):
        raise ValueError("Require 0 <= oversold < overbought <= 100")

    # Compute RSI (Wilder)
    rsi = _rsi_wilder(close, rsi_period)

    n = len(close)
    if n == 0:
        return {"ohlcv": pd.Series(dtype=int)}

    # Build positions statefully to prevent double entries and ensure clean transitions
    pos = np.zeros(n, dtype=int)
    state = 0  # 0 = flat, 1 = long

    rsi_arr = rsi.to_numpy()

    # Iterate through time; decisions at time i use rsi[i-1] and rsi[i] (no future data)
    for i in range(1, n):
        prev_rsi = rsi_arr[i - 1]
        curr_rsi = rsi_arr[i]

        # If RSI not available for either prev or curr, keep current state
        if np.isnan(prev_rsi) or np.isnan(curr_rsi):
            pos[i] = state
            continue

        if state == 0:
            # Entry: prev >= oversold and curr < oversold (cross below)
            if prev_rsi >= oversold and curr_rsi < oversold:
                state = 1
        else:
            # Exit: prev <= overbought and curr > overbought (cross above)
            if prev_rsi <= overbought and curr_rsi > overbought:
                state = 0

        pos[i] = state

    position_series = pd.Series(pos, index=close.index, name="position").astype(int)

    return {"ohlcv": position_series}
