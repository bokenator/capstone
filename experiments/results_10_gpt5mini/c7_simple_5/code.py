from typing import Dict, Any

import numpy as np
import pandas as pd


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
    # Validate input structure
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' mapping to a DataFrame")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = df["close"].astype(float).copy()

    # Read and validate params (use only allowed params)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period)
    # 1) price changes
    delta = close.diff()

    # 2) gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # 3) smoothed averages (Wilder's RSI)
    # Use EWM with adjust=False which is equivalent to Wilder's smoothing
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # 4) Relative strength and RSI
    # Handle cases to avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where both avg_gain and avg_loss are zero (no price movement), set RSI to 50
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi[both_zero] = 50.0

    # Where avg_loss is zero (no losses), RSI = 100
    only_loss_zero = (avg_loss == 0.0) & (~both_zero)
    rsi[only_loss_zero] = 100.0

    # Where avg_gain is zero (no gains), RSI = 0
    only_gain_zero = (avg_gain == 0.0) & (~both_zero)
    rsi[only_gain_zero] = 0.0

    # Mask initial warmup period to NaN to avoid false signals before enough data
    # This ensures the first meaningful RSI value appears at index >= rsi_period
    rsi.iloc[:rsi_period] = np.nan

    # Generate crossing signals
    prev_rsi = rsi.shift(1)

    entries = (prev_rsi >= oversold) & (rsi < oversold)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series without NaN
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series (stateful, long-only)
    n = len(close)
    pos_array = np.zeros(n, dtype=int)
    state = 0  # 0 = flat, 1 = long

    entries_vals = entries.to_numpy(dtype=bool)
    exits_vals = exits.to_numpy(dtype=bool)

    for i in range(n):
        if state == 0:
            if entries_vals[i]:
                state = 1
        else:
            if exits_vals[i]:
                state = 0
        pos_array[i] = state

    position = pd.Series(pos_array, index=close.index, name="position")

    return {"ohlcv": position}
