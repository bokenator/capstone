import pandas as pd
import numpy as np
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Wilder RSI for a price series.

    Uses exponential smoothing with alpha=1/period (Wilder's smoothing).
    Returns a Series aligned with `close` and filled with neutral value 50
    where RSI cannot be computed.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing (alpha = 1/period). Use adjust=False to prevent lookahead.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss == 0 -> RSI = 100 (if avg_gain > 0),
    # avg_gain == 0 -> RSI = 0, both zero -> neutral 50.
    rsi = rsi.where(~(avg_loss == 0), 100.0)
    rsi = rsi.where(~(avg_gain == 0), 0.0)

    # If both avg_gain and avg_loss are zero, set to 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # Fill any remaining NaNs (e.g., first bars before min_periods) with 50 (neutral)
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
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
    # Validate input container type: allow passing the DataFrame directly for convenience
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict) and "ohlcv" in data and isinstance(data["ohlcv"], pd.DataFrame):
        df = data["ohlcv"]
    else:
        raise ValueError("`data` must be a dict with key 'ohlcv' mapping to a DataFrame, or a DataFrame")

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    # Extract and validate params (use only allowed params)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic bounds checking according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    close = df["close"].astype(float).copy()

    # Compute RSI (no lookahead - uses past data only)
    rsi = _compute_rsi(close, rsi_period)

    # Define cross events
    prev_rsi = rsi.shift(1)

    entry_signals = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signals = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series using a deterministic forward-scan to avoid double entries
    position = pd.Series(0, index=close.index, dtype="int8")
    in_position = 0

    # Iterate over index to update position state
    for idx in close.index:
        if in_position == 0:
            if entry_signals.loc[idx]:
                in_position = 1
        else:  # in_position == 1
            if exit_signals.loc[idx]:
                in_position = 0
        position.loc[idx] = in_position

    # Ensure positions are 0 or 1 and have no NaN values
    position = position.fillna(0).astype(int)

    # Return as dict mapping slot name to position series
    return {"ohlcv": position}
