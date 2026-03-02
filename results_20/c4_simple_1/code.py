# RSI Mean Reversion Signal Generator
# Implements generate_signals(...) as required by the prompt

from typing import Dict

import numpy as np
import pandas as pd


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
        raise ValueError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close: pd.Series = df["close"].astype(float).copy()

    # Extract and validate params (use defaults similar to PARAM_SCHEMA)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold level must be less than overbought level")

    def _compute_rsi(close_s: pd.Series, period: int) -> pd.Series:
        """Compute RSI using simple rolling averages (Wilder's smoothing not required).

        Returns a Series indexed like close_s with values in [0, 100] or NaN for warmup.
        """
        delta = close_s.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)

        # Use simple moving average over 'period' for initial RSI
        roll_up = up.rolling(window=period, min_periods=period).mean()
        roll_down = down.rolling(window=period, min_periods=period).mean()

        # Avoid division by zero: handle cases explicitly
        rs = roll_up / roll_down

        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Where both rolls are zero (no price movement), set RSI to 50 (neutral)
        both_zero = (roll_up == 0) & (roll_down == 0)
        rsi.loc[both_zero] = 50.0

        # If roll_down is zero but roll_up > 0 => RSI = 100
        down_zero = (roll_down == 0) & ~both_zero
        rsi.loc[down_zero] = 100.0

        # If roll_up is zero but roll_down > 0 => RSI = 0
        up_zero = (roll_up == 0) & ~both_zero
        rsi.loc[up_zero] = 0.0

        # Clip to 0-100 and keep dtype float
        rsi = rsi.clip(lower=0.0, upper=100.0)
        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Detect crossunder (enter long) and crossover (exit long)
    prev_rsi = rsi.shift(1)

    # Entry: previous RSI > oversold and current RSI <= oversold (crosses below oversold)
    entries = (prev_rsi > oversold) & (rsi <= oversold)

    # Exit: previous RSI < overbought and current RSI >= overbought (crosses above overbought)
    exits = (prev_rsi < overbought) & (rsi >= overbought)

    # Replace NaNs (from warmup) with False for entry/exit booleans
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Convert boolean entry/exit arrays into a position series (0 or 1)
    entries_arr = entries.to_numpy(dtype=bool)
    exits_arr = exits.to_numpy(dtype=bool)

    pos_arr = np.zeros(len(close), dtype=np.int8)
    long = False

    for i in range(len(close)):
        if not long and entries_arr[i]:
            long = True
        elif long and exits_arr[i]:
            long = False
        pos_arr[i] = 1 if long else 0

    position = pd.Series(pos_arr, index=close.index, name="position")

    return {"ohlcv": position}
