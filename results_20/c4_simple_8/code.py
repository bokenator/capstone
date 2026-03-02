import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict


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
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Validate and extract parameters (only allowed params)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Extract with defaults (in case caller omitted some keys)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate ranges according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if not (oversold < overbought):
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI using vectorbt's RSI indicator
    # vbt.RSI.run returns an object with attribute `rsi` (Series or DataFrame)
    rsi_res = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_res.rsi

    # If rsi is a DataFrame with a single column (e.g., when input was 1D), convert to Series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] >= 1:
            # take the first column
            rsi = rsi.iloc[:, 0]
        else:
            # fallback
            rsi = rsi.squeeze()

    # Ensure alignment with close index
    rsi = rsi.reindex(close.index)

    # Detect crossings: require previous value to be non-NaN and comparison with thresholds
    prev = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and now < oversold
    entries = (prev >= oversold) & (rsi < oversold) & prev.notna() & rsi.notna()

    # Exit when RSI crosses above overbought: prev <= overbought and now > overbought
    exits = (prev <= overbought) & (rsi > overbought) & prev.notna() & rsi.notna()

    # Build position series (0 or 1) by walking through time
    idx = close.index
    n = len(idx)
    entries_arr = entries.to_numpy(dtype=bool)
    exits_arr = exits.to_numpy(dtype=bool)

    pos_arr = np.zeros(n, dtype=int)
    in_pos = False

    for i in range(n):
        if not in_pos:
            if entries_arr[i]:
                in_pos = True
                pos_arr[i] = 1
            else:
                pos_arr[i] = 0
        else:
            if exits_arr[i]:
                in_pos = False
                pos_arr[i] = 0
            else:
                pos_arr[i] = 1

    position = pd.Series(pos_arr, index=idx, name="position")

    return {"ohlcv": position}
