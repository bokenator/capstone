import pandas as pd
import numpy as np
from typing import Dict, Any


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
    """
    # Support flexible input: allow passing a DataFrame/Series directly for tests
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = {"ohlcv": pd.DataFrame(data)}

    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    df = data["ohlcv"]

    if "close" not in df.columns:
        raise KeyError("'ohlcv' DataFrame must contain 'close' column")

    close = df["close"].astype(float).copy()

    # Validate and extract params
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        rsi_period = 14

    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Sanity bounds (respect PARAM_SCHEMA)
    rsi_period = max(2, min(100, rsi_period))
    oversold = max(0.0, min(50.0, oversold))
    overbought = max(50.0, min(100.0, overbought))

    # --- RSI calculation (Wilder's smoothing via ewm, causal) ---
    # delta
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha=1/period and adjust=False for Wilder smoothing
    # This is causal: value at t depends only on <= t
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Avoid division by zero
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # rsi may contain NaNs for initial periods; that's fine. We'll treat NaN as no signal.

    # --- Generate entry/exit signals using crossing logic (no lookahead) ---
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN boolean with False
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # --- Build position series (stateful, long-only) ---
    n = len(close)
    pos = np.zeros(n, dtype=np.int8)

    # Convert to numpy arrays for speed
    entry_arr = entry_signal.to_numpy(dtype=bool)
    exit_arr = exit_signal.to_numpy(dtype=bool)

    for i in range(n):
        if i == 0:
            prev = 0
        else:
            prev = int(pos[i - 1])

        cur = prev

        # Enter only if currently flat
        if entry_arr[i] and prev == 0:
            cur = 1
        # Exit only if currently long
        elif exit_arr[i] and prev == 1:
            cur = 0

        pos[i] = cur

    position_series = pd.Series(pos, index=close.index, name="position")

    return {"ohlcv": position_series}
