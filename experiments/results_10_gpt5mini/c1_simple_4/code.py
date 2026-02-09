"""
RSI Mean Reversion Signal Generator

Implements RSI calculation (Wilder's smoothing via EWM) and produces a long-only
position series that goes long when RSI crosses below the oversold threshold and
exits when RSI crosses above the overbought threshold.

Only accesses the 'close' column from the 'ohlcv' DataFrame and only reads the
parameters declared in PARAM_SCHEMA: 'rsi_period', 'oversold', 'overbought'.
"""

from typing import Dict

import numpy as np
import pandas as pd


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
    # --- Validate input data ---
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to pandas DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("'close' column is required in data['ohlcv']")

    close = ohlcv["close"].astype(float).copy()

    # If there's no data, return an empty positions series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int8")}

    # --- Validate and extract parameters ---
    # Only use parameters declared in PARAM_SCHEMA: rsi_period, oversold, overbought
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30.0)
    overbought = params.get("overbought", 70.0)

    # Basic type coercion and validation
    try:
        rsi_period = int(rsi_period)
    except Exception:
        raise TypeError("rsi_period must be an integer")

    try:
        oversold = float(oversold)
        overbought = float(overbought)
    except Exception:
        raise TypeError("oversold and overbought must be numeric (float)")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")

    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # --- Compute RSI using Wilder's smoothing (EWM with alpha=1/period) ---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Use EWM with alpha=1/period and min_periods=period to emulate Wilder's RMA
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss == 0 => RSI = 50 (no price movement)
    rsi = rsi.copy()  # ensure we don't modify underlying series unintentionally

    mask_loss_zero = (avg_loss == 0)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)

    rsi.loc[mask_loss_zero] = 100.0
    rsi.loc[mask_both_zero] = 50.0

    # Keep NaNs for warmup periods

    # --- Generate entry and exit signals based on RSI crosses ---
    prev_rsi = rsi.shift(1)

    # Require both current and previous RSI to be non-NaN to consider a cross
    valid_cross = rsi.notna() & prev_rsi.notna()

    entry_signals = (rsi < oversold) & (prev_rsi >= oversold) & valid_cross
    exit_signals = (rsi > overbought) & (prev_rsi <= overbought) & valid_cross

    # --- Build position series (long-only) ---
    n = len(close)
    positions = np.zeros(n, dtype="int8")

    in_position = False
    # Use iloc-based access for performance and alignment
    entry_vals = entry_signals.values
    exit_vals = exit_signals.values

    for i in range(n):
        if entry_vals[i] and not in_position:
            in_position = True
        elif exit_vals[i] and in_position:
            in_position = False
        positions[i] = 1 if in_position else 0

    position_series = pd.Series(positions, index=close.index, name="position")

    return {"ohlcv": position_series}
