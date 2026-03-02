from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
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
    """

    # Validate input presence
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Extract and validate params with safe defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold threshold must be less than overbought threshold")

    def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
        """
        Compute Wilder's RSI (the classic RSI) using an explicit loop to ensure
        determinism and no lookahead. The first RSI value is produced at index
        'period' (so indices < period are NaN).
        """
        s = series
        n = len(s)
        if n == 0:
            return pd.Series(dtype=float, index=s.index)

        delta = s.diff()
        # Replace NaN delta at first position with 0 for gain/loss calculation
        gain = delta.clip(lower=0).fillna(0.0).to_numpy()
        loss = (-delta.clip(upper=0)).fillna(0.0).to_numpy()

        rsi = np.full(n, np.nan, dtype=float)

        # Need at least period + 1 data points to compute first RSI (since delta starts at index 1)
        if n <= period:
            return pd.Series(rsi, index=s.index)

        # Initial average gain/loss: mean of first `period` gains/losses (from index 1..period)
        # Note: gain[0] corresponds to delta at index 0 which is NaN-filled as 0
        initial_avg_gain = np.mean(gain[1 : period + 1])
        initial_avg_loss = np.mean(loss[1 : period + 1])

        avg_gain = np.full(n, np.nan, dtype=float)
        avg_loss = np.full(n, np.nan, dtype=float)

        avg_gain[period] = initial_avg_gain
        avg_loss[period] = initial_avg_loss

        # Wilder's smoothing for subsequent values
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

        # Compute RSI values where averages are available
        for i in range(period, n):
            ag = avg_gain[i]
            al = avg_loss[i]
            if np.isnan(ag) or np.isnan(al):
                rsi[i] = np.nan
            else:
                if al == 0.0:
                    # If there is no loss, RSI is 100 when there is some gain,
                    # and 50 when both gains and losses are zero (flat)
                    if ag == 0.0:
                        rsi[i] = 50.0
                    else:
                        rsi[i] = 100.0
                else:
                    rs = ag / al
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return pd.Series(rsi, index=s.index)

    rsi_series = _wilder_rsi(close, rsi_period)

    # Define crossing events: crossing is determined using previous value (no lookahead)
    prev_rsi = rsi_series.shift(1)

    cross_below = (prev_rsi >= oversold) & (rsi_series < oversold)
    cross_above = (prev_rsi <= overbought) & (rsi_series > overbought)

    # Build position series iteratively to avoid double entries/exits
    n = len(close)
    positions = np.zeros(n, dtype=int)
    current_pos = 0

    for i in range(n):
        if cross_below.iat[i] and current_pos == 0:
            current_pos = 1
        elif cross_above.iat[i] and current_pos == 1:
            current_pos = 0
        positions[i] = current_pos

    position_series = pd.Series(positions, index=close.index, name="position")

    return {"ohlcv": position_series}
