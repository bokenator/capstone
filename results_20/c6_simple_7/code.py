"""
RSI Mean Reversion Signal Generator

Implements generate_signals(data: dict, params: dict) -> dict
- Computes RSI (Wilder smoothing / exponential) with period params['rsi_period']
- Entry when RSI crosses below params['oversold']
- Exit when RSI crosses above params['overbought']
- Returns positions series (0 = flat, 1 = long) under key 'ohlcv'

The implementation is lookahead-safe and deterministic. It avoids external
dependencies beyond pandas/numpy so it works even if vectorbt is not present.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (exponential moving average).

    Args:
        close: Price series.
        period: RSI lookback period (e.g., 14).

    Returns:
        pd.Series of RSI values aligned with `close` index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing (alpha = 1/period) -> ewm with adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss

    # Compute RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 and avg_gain == 0 -> set RSI = 50 (no movement)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # Where avg_loss == 0 (but avg_gain > 0) -> RSI = 100
    rsi = rsi.where(~(avg_loss == 0) | (avg_gain == 0), 100.0)

    # Keep original index
    rsi.index = close.index
    rsi.name = "rsi"
    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position series based on RSI mean-reversion rules.

    Args:
        data: Dict with key "ohlcv" containing a DataFrame with a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' containing a pd.Series of positions (0 or 1),
        indexed the same as the input close series.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

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

    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    n = len(close)

    # Short-circuit for empty series
    if n == 0:
        return {"ohlcv": pd.Series(dtype=int)}

    # Compute RSI (lookback-only, no future data)
    rsi = _compute_rsi(close, rsi_period)

    # Detect crosses: use previous value (shifted) to avoid lookahead
    prev = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and curr < oversold
    entries = (prev >= oversold) & (rsi < oversold)
    # Exit when RSI crosses above overbought: prev <= overbought and curr > overbought
    exits = (prev <= overbought) & (rsi > overbought)

    # Replace NaNs in boolean arrays with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series (0 = flat, 1 = long) using a single forward scan
    pos_arr = np.zeros(n, dtype=np.int8)
    state = 0  # 0 = flat, 1 = long

    # Iterate sequentially to ensure no lookahead bias
    for i in range(n):
        if state == 0:
            if entries.iloc[i]:
                state = 1
        else:  # state == 1
            if exits.iloc[i]:
                state = 0
        pos_arr[i] = state

    position = pd.Series(pos_arr.astype(int), index=close.index, name="position")

    return {"ohlcv": position}
