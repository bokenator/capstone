import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with
              DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Long-only: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """

    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] DataFrame must contain a 'close' column")

    close = df["close"].astype(float).copy()

    # Extract parameters with defaults/validation per PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if rsi_period > 100:
        raise ValueError("rsi_period must be <= 100")

    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Ensure index is monotonic and preserved
    n = len(close)
    if n == 0:
        return {"ohlcv": pd.Series(dtype=float)}

    # Compute RSI (classic SMA-based implementation, no future data used)
    # 1) Price differences
    delta = close.diff()

    # 2) Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # 3) Average gains/losses using rolling mean (uses only past data)
    # min_periods ensures we don't compute RSI until we have enough data
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    # 4) Compute RSI
    # Handle division by zero safely
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss == 0 and avg_gain > 0 -> RSI = 100
    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~mask_loss_zero, 100.0)

    # When both avg_gain and avg_loss == 0 -> RSI = 50 (no movement)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~mask_both_zero, 50.0)

    # Note: rsi will be NaN for the initial warmup period (first rsi_period-1 bars)

    # Generate position series by simulating the state machine (no lookahead)
    pos = np.zeros(n, dtype=int)
    in_position = False

    # Use numpy arrays for speed
    rsi_values = rsi.values

    # Iterate through time, starting at index 1 since we need previous value to detect cross
    for i in range(1, n):
        prev = rsi_values[i - 1]
        curr = rsi_values[i]

        # Initialize pos[i] to previous position by default
        pos[i] = pos[i - 1]

        # Only evaluate crossings if both previous and current RSI are not NaN
        if not np.isnan(prev) and not np.isnan(curr):
            if not in_position:
                # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
                if (prev >= oversold) and (curr < oversold):
                    in_position = True
                    pos[i] = 1
            else:
                # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
                if (prev <= overbought) and (curr > overbought):
                    in_position = False
                    pos[i] = 0
                else:
                    # remain in position
                    pos[i] = 1
        else:
            # When RSI is NaN (warmup), keep previous position (default 0)
            pos[i] = pos[i - 1]

    # Ensure first bar position is set (default flat)
    pos[0] = 0

    position_series = pd.Series(pos, index=close.index, dtype=int)

    # Final safety checks: only 0 or 1 values
    position_series = position_series.clip(lower=0, upper=1)

    return {"ohlcv": position_series}
