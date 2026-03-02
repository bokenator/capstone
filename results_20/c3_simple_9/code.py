# -*- coding: utf-8 -*-
from typing import Any, Dict

import numpy as np
import pandas as pd


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Wilder's RSI (the classic RSI) without lookahead.

    This implementation uses the standard Wilder smoothing (recursive average)
    seeded by the simple moving average of the first `period` deltas.

    Args:
        close: Close price series.
        period: RSI lookback period (e.g., 14).

    Returns:
        pd.Series: RSI values aligned with `close` index. Values before
        the first valid RSI are NaN.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    close = close.astype(float)
    length = len(close)

    # Prepare output
    rsi_values = np.full(length, np.nan)

    if length < period + 1:
        return pd.Series(rsi_values, index=close.index)

    # Price changes
    delta = close.diff().values  # delta[0] is nan

    # Gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Arrays to hold smoothed averages
    avg_gain = np.full(length, np.nan)
    avg_loss = np.full(length, np.nan)

    # Seed the first average using simple mean of first `period` deltas
    # We use deltas 1..period inclusive
    start_idx = period
    seed_gain = gains[1 : period + 1].mean()
    seed_loss = losses[1 : period + 1].mean()

    avg_gain[start_idx] = seed_gain
    avg_loss[start_idx] = seed_loss

    # Wilder's smoothing (recursive)
    for i in range(start_idx + 1, length):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    # Compute RSI where averages are available
    for i in range(start_idx, length):
        ag = avg_gain[i]
        al = avg_loss[i]
        if al == 0.0:
            # If both are zero, RSI is neutral (50); if only loss is zero -> RSI=100
            rsi_values[i] = 50.0 if ag == 0.0 else 100.0
        else:
            rs = ag / al
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi_values, index=close.index)


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy:
    - Compute RSI with period `rsi_period`.
    - Enter long (1) when RSI crosses below `oversold` (e.g., 30).
    - Exit to flat (0) when RSI crosses above `overbought` (e.g., 70).

    Args:
        data: Either a dict with key "ohlcv" containing a DataFrame with
              a 'close' column, or directly a DataFrame (for convenience).
        params: Dictionary with keys:
            - 'rsi_period' (int)
            - 'oversold' (float)
            - 'overbought' (float)

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1)
        aligned with the input close index.
    """
    # Accept both dict-with-ohlcv and raw DataFrame for flexibility in tests
    ohlcv = None
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        ohlcv = data.get("ohlcv")
        if ohlcv is None:
            raise ValueError("data dict must contain 'ohlcv' DataFrame")
    else:
        raise TypeError("data must be a DataFrame or a dict containing 'ohlcv'")

    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].copy()

    # Extract parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI (no lookahead)
    rsi = _wilder_rsi(close, rsi_period)

    # Detect crossings using previous bar values (no future info)
    prev_rsi = rsi.shift(1)

    entry_mask = (rsi < oversold) & (prev_rsi >= oversold)
    exit_mask = (rsi > overbought) & (prev_rsi <= overbought)

    n = len(close)
    positions = np.zeros(n, dtype=np.int8)

    # Iterate forward to build position series (ensures we don't double-enter)
    for i in range(n):
        if i == 0:
            positions[i] = 0
            continue
        positions[i] = positions[i - 1]

        # Only enter if flat
        if positions[i - 1] == 0 and entry_mask.iloc[i]:
            positions[i] = 1
        # Only exit if long
        elif positions[i - 1] == 1 and exit_mask.iloc[i]:
            positions[i] = 0

    position_series = pd.Series(positions, index=close.index, name="position")

    # Ensure dtype is numeric and no NaNs (fill any accidental NaNs with 0)
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
