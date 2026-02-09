import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with a DataFrame
              that has a 'close' column. For convenience, a DataFrame may be passed directly
              (it will be treated as the 'ohlcv' DataFrame).
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY strategy: values are 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """

    # Accept either a dict with 'ohlcv' or a DataFrame directly (for testing convenience)
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data dictionary must contain 'ohlcv' key with a DataFrame")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("data must be a pandas DataFrame or a dict containing 'ohlcv' DataFrame")

    # Validate close column
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    # Parameters with sensible defaults if keys missing
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    oversold = float(params.get("oversold", 30.0)) if params is not None else 30.0
    overbought = float(params.get("overbought", 70.0)) if params is not None else 70.0

    # Defensive checks on params
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    close = ohlcv["close"].astype(float)

    # If close is empty, return empty position series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int64")}

    # Compute RSI using Wilder's smoothing (causal - depends only on current and past data)
    # RSI = 100 * avg_gain / (avg_gain + avg_loss)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via exponential moving average with alpha = 1/period
    # adjust=False ensures the ewm is recursive (Wilder-style)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    denom = avg_gain + avg_loss
    # Compute RSI safely: where denom == 0 set neutral 50, otherwise compute fraction
    rsi = pd.Series(np.nan, index=close.index, name="rsi")
    non_zero = denom != 0
    rsi.loc[non_zero] = 100.0 * (avg_gain.loc[non_zero] / denom.loc[non_zero])
    # For points where both avg_gain and avg_loss are zero (flat series), set RSI to 50 (neutral)
    rsi.loc[~non_zero] = 50.0

    # Detect crosses (use previous RSI value to avoid lookahead)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses BELOW the oversold threshold (prev >= oversold and curr < oversold)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE the overbought threshold (prev <= overbought and curr > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position time series (0 = flat, 1 = long) using a simple state machine to avoid
    # double entries/exits and ensure determinism. This loop is causal (uses only past data).
    n = len(close)
    pos = np.zeros(n, dtype="int8")

    # Use integer positions and preserve the index
    for i in range(n):
        if i == 0:
            pos[i] = 0
            continue

        prev_pos = pos[i - 1]

        # Only consider entry if currently flat
        if prev_pos == 0 and bool(entry_mask.iloc[i]):
            pos[i] = 1
            continue

        # Only consider exit if currently long
        if prev_pos == 1 and bool(exit_mask.iloc[i]):
            pos[i] = 0
            continue

        # Otherwise carry forward previous position
        pos[i] = prev_pos

    position_series = pd.Series(pos, index=close.index, name="position").astype(int)

    # Ensure no NaNs (after warmup we must have concrete positions). Fill any remaining NaN with 0.
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
