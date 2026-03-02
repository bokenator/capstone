# RSI Mean Reversion Signal Generator
from typing import Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI period (int).

    Returns:
        RSI series (float) aligned with close.index.
    """
    if period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via ewm with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Replace infinite RSI (when avg_loss == 0) with 100, and fill initial NaNs with neutral 50
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: dict,
) -> Dict[str, pd.Series]:
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
    """
    # Basic validation
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot name to DataFrame (contains 'ohlcv')")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Read and validate params (only allowed params)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Prepare values for crossing detection. Fill NaNs with neutral value (50)
    rsi_filled = rsi.fillna(50.0)
    rsi_prev = rsi_filled.shift(1).fillna(50.0)

    # Entry: RSI crosses BELOW oversold (previous >= oversold, current < oversold)
    entry = (rsi_filled < oversold) & (rsi_prev >= oversold)

    # Exit: RSI crosses ABOVE overbought (previous <= overbought, current > overbought)
    exit_ = (rsi_filled > overbought) & (rsi_prev <= overbought)

    # Build position series (0 = flat, 1 = long). Ensure no lookahead by iterating chronologically.
    pos_array = np.zeros(len(close), dtype=int)
    in_long = 0

    entry_vals = entry.values
    exit_vals = exit_.values

    for i in range(len(close)):
        if in_long == 0 and entry_vals[i]:
            in_long = 1
        elif in_long == 1 and exit_vals[i]:
            in_long = 0
        pos_array[i] = in_long

    position = pd.Series(pos_array, index=close.index)

    return {"ohlcv": position}
