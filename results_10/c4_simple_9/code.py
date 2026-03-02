import numpy as np
import pandas as pd
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of closing prices.
        period: RSI period (int).

    Returns:
        RSI as a pd.Series (same index as close). NaN may appear for initial bars.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - 100 / (1 + rs)

    # Handle the case where both avg_gain and avg_loss are 0 -> no movement -> RSI = 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    if both_zero.any():
        rsi = rsi.copy()
        rsi.loc[both_zero] = 50.0

    return rsi


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
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
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to pd.DataFrame")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = df["close"].astype(float).copy()

    # Extract and validate parameters
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")

    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    if not (oversold < overbought):
        raise ValueError("oversold must be less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect cross events
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs in boolean masks with False to avoid accidental signals
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series (1 = long, 0 = flat). Iterate chronologically.
    pos_array = np.zeros(len(close), dtype=int)
    current_pos = 0

    entry_vals = entries.values
    exit_vals = exits.values

    for i in range(len(close)):
        # Only allow long entries if currently flat
        if entry_vals[i] and current_pos == 0:
            current_pos = 1
        # Only exit if currently long
        elif exit_vals[i] and current_pos == 1:
            current_pos = 0
        pos_array[i] = current_pos

    positions = pd.Series(pos_array, index=close.index, name="position")

    return {"ohlcv": positions}
