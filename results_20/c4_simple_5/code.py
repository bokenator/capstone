import pandas as pd
import numpy as np
from typing import Dict


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute Wilder's RSI for a close price series.

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series (float) with same index as close. Values are NaN until enough data.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    close = close.astype(float)
    delta = close.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # First average gain/loss: simple moving average over first 'period' values
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rsi = pd.Series(index=close.index, dtype=float)

    # Find first index where rolling averages are valid
    first_valid = avg_gain.first_valid_index()
    if first_valid is None:
        # Not enough data to compute RSI
        return rsi

    first_idx = close.index.get_loc(first_valid)

    # Prepare arrays for Wilder smoothing
    n = len(close)
    avg_gain_s = np.full(n, np.nan, dtype=float)
    avg_loss_s = np.full(n, np.nan, dtype=float)

    avg_gain_s[first_idx] = avg_gain.iloc[first_idx]
    avg_loss_s[first_idx] = avg_loss.iloc[first_idx]

    # Wilder's smoothing for subsequent values
    for i in range(first_idx + 1, n):
        g = gain.iloc[i] if not pd.isna(gain.iloc[i]) else 0.0
        l = loss.iloc[i] if not pd.isna(loss.iloc[i]) else 0.0
        avg_gain_s[i] = (avg_gain_s[i - 1] * (period - 1) + g) / period
        avg_loss_s[i] = (avg_loss_s[i - 1] * (period - 1) + l) / period

    # Compute RSI
    rsi_vals = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(avg_gain_s[i]) or np.isnan(avg_loss_s[i]):
            continue
        if avg_loss_s[i] == 0.0:
            # Prevent division by zero: if no losses then RSI is 100
            rsi_vals[i] = 100.0
        else:
            rs = avg_gain_s[i] / avg_loss_s[i]
            rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))

    rsi[:] = rsi_vals
    return rsi


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
    # Validate data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = df["close"].astype(float)

    # Validate and extract params (use only allowed params)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if not (oversold < overbought):
        raise ValueError("oversold must be less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Entry: RSI crosses BELOW oversold (prev >= oversold -> now < oversold)
    prev_rsi = rsi.shift(1)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE overbought (prev <= overbought -> now > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean Series aligned with close index and no NaNs
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Simulate long-only position: 1 when long, 0 when flat
    n = len(close)
    pos_values = np.zeros(n, dtype=int)
    long = False

    for i in range(n):
        if entry_mask.iloc[i] and not long:
            long = True
        elif exit_mask.iloc[i] and long:
            long = False
        pos_values[i] = 1 if long else 0

    position = pd.Series(pos_values, index=close.index)

    return {"ohlcv": position}
