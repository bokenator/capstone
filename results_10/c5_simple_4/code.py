from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """

    # Basic validations
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("'ohlcv' key missing in data")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("'close' column missing in data['ohlcv']")

    # Read and sanitize price series
    close: pd.Series = ohlcv["close"].astype(float).copy()
    # Replace infinities with NaN, then forward-fill to handle intermittent NaNs without lookahead
    close = close.replace([np.inf, -np.inf], np.nan).ffill()

    # Parameters with defaults and validation according to PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    oversold = float(params.get("oversold", 30.0)) if params is not None else 30.0
    overbought = float(params.get("overbought", 70.0)) if params is not None else 70.0

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Helper: compute RSI using Wilder's smoothing (causal: uses only past data)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        # Gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing via ewm with alpha=1/period and adjust=False (recursive)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # If avg_loss == 0 --> RSI = 100
        rsi = rsi.where(~(avg_loss == 0), 100.0)
        # If both avg_gain and avg_loss == 0 --> RSI = 50 (no movement)
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~both_zero, 50.0)

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Determine crosses using previous bar (no lookahead)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs (caused by warmup) with False for signal arrays
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build the position series (stateful: once long, stay long until exit)
    n = len(rsi)
    pos = np.zeros(n, dtype=int)
    in_position = 0

    # Use .iat for fast scalar access
    for i in range(n):
        if entry_signals.iat[i] and in_position == 0:
            in_position = 1
        elif exit_signals.iat[i] and in_position == 1:
            in_position = 0
        pos[i] = in_position

    position_series = pd.Series(pos, index=close.index, name="position").astype(int)

    return {"ohlcv": position_series}
