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
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period
            - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
            - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY: values are 1 (long) or 0 (flat).
    """
    # Validate input container
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict mapping slot name to a DataFrame (contains 'ohlcv')")

    if "ohlcv" not in data:
        raise ValueError("`data` must contain 'ohlcv' key with a DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`data['ohlcv']` must be a pandas DataFrame")

    if "close" not in df.columns:
        raise ValueError("`data['ohlcv']` must contain 'close' column")

    close = df["close"].astype(float)

    # Extract and validate params
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing via ewm with alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

        denom = avg_gain + avg_loss
        # Initialize RSI with 50 as neutral where not computable
        rsi = pd.Series(50.0, index=series.index)
        nonzero = denom != 0
        rsi.loc[nonzero] = 100.0 * (avg_gain.loc[nonzero] / denom.loc[nonzero])

        # Keep dtype float
        return rsi

    rsi = _compute_rsi(close, rsi_period)

    n = len(close)
    positions = np.zeros(n, dtype=int)

    # Sequential state-machine to avoid double entries/exits and ensure no lookahead
    # Position values: 0 = flat, 1 = long
    for i in range(1, n):
        prev_pos = positions[i - 1]
        # Default carry forward previous position
        positions[i] = prev_pos

        prev_rsi = rsi.iat[i - 1]
        curr_rsi = rsi.iat[i]

        # If RSI values are NaN (not enough data), do nothing
        if pd.isna(prev_rsi) or pd.isna(curr_rsi):
            continue

        # Entry: when RSI crosses BELOW oversold (prev >= oversold and curr < oversold)
        if prev_pos == 0:
            if (prev_rsi >= oversold) and (curr_rsi < oversold):
                positions[i] = 1
                continue

        # Exit: when RSI crosses ABOVE overbought (prev <= overbought and curr > overbought)
        if prev_pos == 1:
            if (prev_rsi <= overbought) and (curr_rsi > overbought):
                positions[i] = 0
                continue

    # Construct pandas Series for positions with same index as input
    position_series = pd.Series(positions, index=close.index, name="position").astype(int)

    # Ensure no NaNs (positions are integers 0/1). This also satisfies the "no NaN after warmup" test
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
