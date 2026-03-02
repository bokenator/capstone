import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Wilder's RSI using exponential smoothing (no lookahead).

    Args:
        close: Series of close prices.
        period: RSI period (integer >= 2).

    Returns:
        RSI series aligned with `close` index. Values are floats in [0,100] or NaN when
        there is not enough data.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    close = close.astype(float)
    delta = close.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's smoothing with alpha = 1/period. Use adjust=False for recursive formula.
    # Use min_periods=period so RSI is NaN until enough data exists.
    alpha = 1.0 / float(period)
    ma_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Avoid division warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases: if ma_down == 0 and ma_up > 0 -> RSI = 100.
    # If both are zero -> RSI = 50 (no moves).
    both_zero = (ma_up == 0) & (ma_down == 0)
    down_zero = (ma_down == 0) & (ma_up != 0)

    rsi = rsi.copy()
    rsi[down_zero] = 100.0
    rsi[both_zero] = 50.0

    return rsi


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
    # Validate input data structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = df["close"]
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=df.index)

    # Extract and validate params, using defaults if missing
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI (no lookahead)
    rsi = _compute_rsi(close, rsi_period)

    # Entry: RSI crosses below oversold (prev >= oversold and current < oversold)
    prev_rsi = rsi.shift(1)
    entry_mask = (rsi < oversold) & (prev_rsi >= oversold)

    # Exit: RSI crosses above overbought (prev <= overbought and current > overbought)
    exit_mask = (rsi > overbought) & (prev_rsi <= overbought)

    # Replace NaNs in masks with False (no signal when RSI is NaN)
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Convert to numpy arrays for a simple finite-state simulation (no lookahead)
    entry_arr = entry_mask.to_numpy(dtype=bool)
    exit_arr = exit_mask.to_numpy(dtype=bool)

    n = len(close)
    pos_values = np.zeros(n, dtype=np.int8)
    in_position = False

    for i in range(n):
        if in_position:
            # If currently long and an exit signal occurs, exit
            if exit_arr[i]:
                in_position = False
        else:
            # If currently flat and an entry signal occurs, enter
            if entry_arr[i]:
                in_position = True
        pos_values[i] = 1 if in_position else 0

    position = pd.Series(pos_values, index=close.index, name="position")

    return {"ohlcv": position}
