from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series aligned with input index. Values may be NaN for the initial periods (< period).
    """
    # Ensure float dtype and copy to avoid modifying original
    close = close.astype("float64").copy()

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing using ewm with alpha=1/period (adjust=False -> recursive)
    # min_periods=period ensures we don't produce values before enough data
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 and avg_gain > 0 -> RSI should be 100
    # - If avg_loss == 0 and avg_gain == 0 -> RSI should be 50 (no change)
    mask_loss_zero = (avg_loss == 0)
    rsi = rsi.where(~(mask_loss_zero & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~(mask_loss_zero & (avg_gain == 0)), 50.0)

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY strategy: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Validate and extract ohlcv DataFrame
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict with key 'ohlcv' mapping to a DataFrame.")

    if "ohlcv" not in data:
        raise ValueError("`data` dict must contain the key 'ohlcv' with a DataFrame value.")

    df = data["ohlcv"]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    # Ensure 'close' column exists (DATA_SCHEMA requirement)
    if "close" not in df.columns:
        # If there is only one column, assume it is the close series
        if df.shape[1] == 1:
            close = df.iloc[:, 0].rename("close")
        else:
            raise ValueError("DataFrame in data['ohlcv'] must contain a 'close' column")
    else:
        close = df["close"].rename("close")

    # Ensure index is preserved
    close = close.astype("float64")

    # Parameters (only allowed keys from PARAM_SCHEMA)
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI (uses only past and present data)
    rsi = _compute_rsi(close, rsi_period)

    # Define entry and exit on RSI crossings (use previous bar to detect crossing)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold, curr < oversold)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold) & prev_rsi.notna() & rsi.notna()

    # Exit: RSI crosses above overbought (prev <= overbought, curr > overbought)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought) & prev_rsi.notna() & rsi.notna()

    # Initialize position series (0 = flat, 1 = long)
    pos = pd.Series(0, index=close.index, dtype="int8")

    # Apply simple state machine to avoid double entries/exits
    in_long = 0
    n = len(pos)

    # Iterate sequentially to ensure no lookahead and correct state transitions
    for i in range(n):
        if in_long:
            # If currently long, check for exit
            if exit_mask.iat[i]:
                in_long = 0
        else:
            # If currently flat, check for entry
            if entry_mask.iat[i]:
                in_long = 1

        pos.iat[i] = in_long

    pos = pos.astype("int8")
    pos.name = "position"

    return {"ohlcv": pos}
