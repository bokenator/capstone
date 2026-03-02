# rsi_mean_reversion.py
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Wilder's RSI for a price series.

    Uses the exponential moving average with alpha = 1/period (Wilder smoothing).

    Args:
        close: Close price series.
        period: RSI lookback period (must be >= 2).

    Returns:
        RSI series (float) with same index as `close`. Values are NaN for the
        warmup period.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Relative Strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    # Handle divisions by zero: if avg_loss == 0 -> RSI = 100; if avg_gain == 0 -> RSI = 0
    rsi = rsi.where(~avg_loss.eq(0), 100.0)
    rsi = rsi.where(~avg_gain.eq(0) | avg_loss.eq(0), 0.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict,
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Strategy:
    - Compute RSI on 'close' with period = params['rsi_period']
    - Enter long when RSI CROSSES BELOW params['oversold']
    - Exit long when RSI CROSSES ABOVE params['overbought']
    - Long-only: positions are 1 (long) or 0 (flat)

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' with 'close'.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1).
    """
    # Validate inputs
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    df = data["ohlcv"]
    if "close" not in df.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = df["close"].astype(float)

    # Extract and validate params (only use allowed params)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period out of allowed range [2, 100]")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Define crossover conditions
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and current < oversold)
    entry_points = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and current > overbought)
    exit_points = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series by scanning through bars
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_long = False

    # If rsi is NaN at a bar, treat signals as False
    entry_points = entry_points.fillna(False)
    exit_points = exit_points.fillna(False)

    for i, idx in enumerate(close.index):
        if in_long:
            if exit_points.iloc[i]:
                in_long = False
                pos.iloc[i] = 0
            else:
                pos.iloc[i] = 1
        else:
            if entry_points.iloc[i]:
                in_long = True
                pos.iloc[i] = 1
            else:
                pos.iloc[i] = 0

    # Ensure position is integer 0 or 1
    pos = pos.astype(int)

    return {"ohlcv": pos}
