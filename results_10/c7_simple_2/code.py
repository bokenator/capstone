import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series (float) with the same index as `close`.
    """
    # Ensure series
    close = close.astype(float)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing: exponential moving average with alpha=1/period and adjust=False
    # This uses only past data (no lookahead)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss is zero (no losses), RSI should be 100. Where avg_gain is zero (no gains), RSI should be 0.
    rsi = rsi.fillna(0.0)
    rsi.loc[avg_loss == 0.0] = 100.0

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
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("`data` must contain 'ohlcv' key with a DataFrame that has a 'close' column")

    df = data["ohlcv"]
    if "close" not in df.columns:
        raise KeyError("`ohlcv` DataFrame must contain 'close' column")

    close = df["close"].copy()

    # Read parameters (only allowed params)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate parameter ranges (defensive)
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect crosses (using previous bar vs current bar)
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses BELOW the oversold threshold (was >= oversold, now < oversold)
    entries = (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE the overbought threshold (was <= overbought, now > overbought)
    exits = (rsi_prev <= overbought) & (rsi > overbought)

    # Ensure boolean dtype and no NaNs
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series (stateful, long-only: 0 or 1)
    position = pd.Series(0, index=close.index, dtype="int8")

    prev_pos = 0
    # Iterate to avoid lookahead and ensure clean entry/exit logic
    for i, idx in enumerate(close.index):
        if prev_pos == 0:
            if entries.iloc[i]:
                curr_pos = 1
            else:
                curr_pos = 0
        else:  # prev_pos == 1
            if exits.iloc[i]:
                curr_pos = 0
            else:
                curr_pos = 1

        position.iloc[i] = curr_pos
        prev_pos = curr_pos

    # Return only the 'ohlcv' slot as required
    return {"ohlcv": position}
