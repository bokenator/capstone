"""
RSI Mean Reversion Signal Generator

Implements generate_signals(...) as specified.

Strategy:
- Compute RSI (Wilder's smoothing via ewm, causal)
- Enter long when RSI crosses BELOW oversold threshold
- Exit long when RSI crosses ABOVE overbought threshold
- Long-only single asset. Returns positions series (0 or 1)

The implementation is defensive and also accepts a bare DataFrame as `data`
for convenience in testing wrappers.
"""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column. For convenience, a bare
              DataFrame may also be passed directly as `data`.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series (0 for flat, 1 for long).
    """

    # Normalize input to obtain the OHLCV DataFrame
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("Input `data` dict must contain 'ohlcv' key with a DataFrame.")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("`data` must be a pandas DataFrame or a dict containing 'ohlcv' DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("Input 'ohlcv' DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Read parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic parameter validation
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Calculate RSI using Wilder's smoothing (EWMA with adjust=False)
    # This implementation is causal (no lookahead).
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use ewm with alpha=1/period (Wilder's smoothing). min_periods ensures
    # RSI is NaN until at least `rsi_period` periods are available.
    alpha = 1.0 / float(rsi_period)

    avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()
    avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()

    # Compute RSI safely handling division-by-zero
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_gain or avg_loss are zero
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    loss_zero = (avg_loss == 0.0) & (avg_gain != 0.0)
    gain_zero = (avg_gain == 0.0) & (avg_loss != 0.0)

    rsi = rsi.where(~both_zero, 50.0)
    rsi = rsi.where(~loss_zero, 100.0)
    rsi = rsi.where(~gain_zero, 0.0)

    # Generate crossing signals (using previous bar to detect crosses)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses BELOW oversold (prev >= oversold and current < oversold)
    entry = (prev_rsi >= oversold) & (rsi < oversold)
    entry = entry.fillna(False)

    # Exit: RSI crosses ABOVE overbought (prev <= overbought and current > overbought)
    exit = (prev_rsi <= overbought) & (rsi > overbought)
    exit = exit.fillna(False)

    # Build position series (0 = flat, 1 = long) - stateful to avoid double entries
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)

    long = False
    # Use positional indexing to be efficient and deterministic
    for i in range(n):
        if not long:
            if bool(entry.iloc[i]):
                long = True
        else:
            if bool(exit.iloc[i]):
                long = False
        pos_arr[i] = 1 if long else 0

    position = pd.Series(pos_arr, index=close.index, name="position")

    return {"ohlcv": position}
