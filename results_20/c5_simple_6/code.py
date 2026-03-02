import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period
            - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
            - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY strategy: 1 (long) or 0 (flat).
    """
    # Accept either a dict with 'ohlcv' or a DataFrame directly for flexibility
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data must contain 'ohlcv' key mapping to a DataFrame")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("data must be a pandas DataFrame or a dict containing 'ohlcv')")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Only forward-fill missing values to avoid lookahead bias (do not backfill)
    close_ffill = close.fillna(method="ffill")

    # Extract parameters (use only allowed params)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 100.0):
        raise ValueError("oversold must be between 0 and 100")
    if not (0.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 0 and 100")

    def compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute RSI using Wilder's smoothing (iterative, no lookahead).

        Returns a Series with same index as input. Values will be NaN for indices
        where RSI cannot be computed (first `period` bars).
        """
        arr = series.to_numpy(dtype=float)
        n = len(arr)
        rsi_arr = np.full(n, np.nan)
        if n <= period:
            return pd.Series(rsi_arr, index=series.index)

        # Calculate deltas
        delta = np.empty(n, dtype=float)
        delta[0] = 0.0
        delta[1:] = arr[1:] - arr[:-1]

        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        # Initial average gain/loss (Wilder's method) at index `period`:
        # average of the first `period` gains/losses (delta[1:period+1])
        initial_gain = gain[1: period + 1].mean()
        initial_loss = loss[1: period + 1].mean()

        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[period] = initial_gain
        avg_loss[period] = initial_loss

        # Wilder smoothing (recursive)
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

        # Compute RSI values from index `period` onwards
        for i in range(period, n):
            if avg_loss[i] == 0.0:
                # If no losses, RSI is 100 (if there were gains) or 50 if no movement
                rsi_arr[i] = 100.0 if avg_gain[i] > 0.0 else 50.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi_arr[i] = 100.0 - (100.0 / (1.0 + rs))

        return pd.Series(rsi_arr, index=series.index)

    rsi_series = compute_rsi(close_ffill, rsi_period)

    # Detect crossings using previous bar (no lookahead)
    prev_rsi = rsi_series.shift(1)
    cross_below = (prev_rsi >= oversold) & (rsi_series < oversold) & prev_rsi.notna() & rsi_series.notna()
    cross_above = (prev_rsi <= overbought) & (rsi_series > overbought) & prev_rsi.notna() & rsi_series.notna()

    n = len(close)
    position = np.zeros(n, dtype=int)
    in_pos = 0

    # Iterate forward in time to build position series (long-only)
    for i in range(n):
        if in_pos == 0:
            if cross_below.iloc[i]:
                in_pos = 1
        else:
            if cross_above.iloc[i]:
                in_pos = 0
        position[i] = in_pos

    position_series = pd.Series(position, index=close.index)

    return {"ohlcv": position_series}
