import pandas as pd
import numpy as np
from typing import Any, Dict


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    This implementation uses an exponential moving average with alpha=1/period
    (adjust=False) which corresponds to Wilder's smoothing. It returns a Series
    aligned with the input 'close' index.
    """
    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via exponential weighted moving average
    # min_periods=period to avoid spurious values in the initial warmup
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Relative strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" containing a DataFrame with a 'close' column.
        params: dict containing 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        dict with key "ohlcv" mapping to a pandas Series of positions (0 or 1),
        indexed the same as the input close prices.
    """
    # Basic validation
    if "ohlcv" not in data:
        raise ValueError("data must contain an 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Handle empty input
    if len(close) == 0:
        return {"ohlcv": pd.Series(dtype=int)}

    # Parameters with safe defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Fill small gaps in price series using forward fill (no lookahead)
    close = close.ffill().bfill()

    # Compute RSI
    rsi = compute_rsi(close, rsi_period)

    # Initialize position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    # Iterate bars sequentially to avoid lookahead and to enforce single-entry logic
    for i in range(len(close)):
        curr_rsi = rsi.iat[i]

        # If RSI is not available for this bar, keep previous position
        if pd.isna(curr_rsi):
            position.iat[i] = 1 if in_position else 0
            continue

        # Entry rule: go long when RSI is below the oversold threshold
        if (not in_position) and (curr_rsi < oversold):
            in_position = True

        # Exit rule: exit long when RSI is above the overbought threshold
        elif in_position and (curr_rsi > overbought):
            in_position = False

        position.iat[i] = 1 if in_position else 0

    return {"ohlcv": position}
