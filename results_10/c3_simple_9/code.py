# RSI Mean Reversion Signal Generator
# Implements: RSI(14) entry when RSI crosses below oversold (30), exit when RSI crosses above overbought (70)

from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" containing a DataFrame with a 'close' column.
              For convenience, the function also accepts a DataFrame (treated as ohlcv)
              or a Series (treated as close prices).
        params: dict with keys:
            - 'rsi_period' (int): period for RSI calculation (default 14)
            - 'oversold' (float): RSI threshold to enter long (default 30.0)
            - 'overbought' (float): RSI threshold to exit long (default 70.0)

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pandas Series
              indexed same as input close and contains 0 (flat) or 1 (long).
    """

    # Validate and extract close series
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict) and "ohlcv" in data:
        ohlcv = data["ohlcv"]
    else:
        raise ValueError("data must be a DataFrame or a dict with key 'ohlcv' containing a DataFrame")

    # If ohlcv is a Series, treat it as close prices
    if isinstance(ohlcv, pd.Series):
        close = ohlcv.copy()
    elif isinstance(ohlcv, pd.DataFrame):
        if "close" not in ohlcv.columns:
            raise ValueError("ohlcv DataFrame must contain a 'close' column")
        close = ohlcv["close"].copy()
    else:
        raise ValueError("ohlcv must be a pandas DataFrame or Series")

    # Ensure float dtype for mathematical operations
    close = close.astype(float)

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Handle trivial cases
    n = len(close)
    if n == 0:
        # Return empty Series with same index
        return {"ohlcv": pd.Series([], dtype="int64")}

    # To avoid lookahead when imputing missing data, only forward-fill NaNs.
    # Do not backfill to prevent using future data.
    close_ffill = close.ffill()

    # Compute RSI (Wilder's smoothing via ewm with alpha=1/period)
    delta = close_ffill.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha = 1/period (Wilder's smoothing)
    # adjust=False ensures recursive one-step smoothing (no lookahead)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Relative strength and RSI
    # Avoid division by zero; where avg_loss == 0 and avg_gain == 0 -> RSI = 50 (no movement)
    # where avg_loss == 0 and avg_gain > 0 -> RSI = 100
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle infinite or NaN cases explicitly
    # If avg_loss == 0 & avg_gain == 0 -> set RSI = 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # If avg_loss == 0 & avg_gain > 0 -> RSI = 100
    rsi = rsi.where(~(avg_loss == 0) | (avg_gain == 0), 100.0)

    # If avg_gain == 0 & avg_loss > 0 -> RSI = 0
    rsi = rsi.where(~(avg_gain == 0) | (avg_loss == 0), 0.0)

    # Ensure index aligns with close
    rsi = rsi.reindex(close.index)

    # Detect cross-under (enter) and cross-over (exit)
    prev_rsi = rsi.shift(1)

    enter_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series iteratively to ensure no double entries/exits
    positions = np.zeros(n, dtype="int64")

    # Fill positions step-by-step
    # positions[0] = 0 (flat) by default
    for i in range(1, n):
        if positions[i - 1] == 0:
            # Only enter if currently flat and enter signal occurs
            if bool(enter_signals.iat[i]):
                positions[i] = 1
            else:
                positions[i] = 0
        else:
            # Currently long: stay long unless an exit signal occurs
            if bool(exit_signals.iat[i]):
                positions[i] = 0
            else:
                positions[i] = 1

    # Convert to pandas Series with same index
    position_series = pd.Series(positions, index=close.index, name="position")

    # Ensure values are strictly 0 or 1 and no NaNs
    position_series = position_series.fillna(0).astype(int)

    return {"ohlcv": position_series}
