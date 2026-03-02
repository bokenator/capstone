import pandas as pd
import numpy as np
from typing import Any


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
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("`data` must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("`data['ohlcv']` must contain a 'close' column")

    # Extract close prices and ensure float dtype
    close: pd.Series = df["close"].astype(float).copy()

    # Read parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate parameter ranges (as per PARAM_SCHEMA)
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold level must be strictly less than overbought level")

    # Handle empty series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int64")}

    # Compute RSI using Wilder's smoothing (approximated with ewm)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha=1/period to approximate Wilder's RMA
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RS and RSI. Handle division-by-zero gracefully.
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # rsi may be NaN where avg_gain and avg_loss are both zero; keep NaN to avoid spurious signals

    # Define cross events: entry when RSI crosses below oversold; exit when RSI crosses above overbought
    prev_rsi = rsi.shift(1)

    # Only consider crosses when both current and previous RSI are finite
    valid_current = rsi.notna()
    valid_prev = prev_rsi.notna()

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold) & valid_current & valid_prev
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought) & valid_current & valid_prev

    # Build position series by iterating through signals to enforce single long position at a time
    pos = np.zeros(len(close), dtype=int)
    current_pos = 0

    # Use integer location-based access for performance
    entry_vals = entry_signals.values
    exit_vals = exit_signals.values

    for i in range(len(close)):
        if entry_vals[i] and current_pos == 0:
            current_pos = 1
        elif exit_vals[i] and current_pos == 1:
            current_pos = 0
        pos[i] = current_pos

    position_series = pd.Series(pos, index=close.index, name="position").astype("int64")

    return {"ohlcv": position_series}
