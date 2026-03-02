import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period
            - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
            - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series (values 0 or 1).
    """
    # Accept either a dict with 'ohlcv' or a DataFrame directly for flexibility in testing
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict) and "ohlcv" in data and isinstance(data["ohlcv"], pd.DataFrame):
        ohlcv = data["ohlcv"]
    else:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame or a DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Extract and validate params (use provided defaults if missing)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use exponential weighted mean with alpha=1/period which corresponds to Wilder's smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Prevent division by zero
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle edge cases explicitly
        # If both avg_gain and avg_loss are zero -> RSI = 50 (no movement)
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        only_loss_zero = (avg_loss == 0) & (avg_gain > 0)
        only_gain_zero = (avg_gain == 0) & (avg_loss > 0)

        rsi = rsi.mask(both_zero, 50.0)
        rsi = rsi.mask(only_loss_zero, 100.0)
        rsi = rsi.mask(only_gain_zero, 0.0)

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Generate crossing signals (no lookahead: use shifted RSI for previous value)
    rsi_prev = rsi.shift(1)

    entry_cond = (rsi_prev >= oversold) & (rsi < oversold)
    exit_cond = (rsi_prev <= overbought) & (rsi > overbought)

    # Build the position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    # Iterate in chronological order to avoid lookahead and to prevent double entries
    for i in range(len(position)):
        if in_position:
            # If currently long, check for exit
            if bool(exit_cond.iloc[i]):
                in_position = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1
        else:
            # If flat, check for entry
            if bool(entry_cond.iloc[i]):
                in_position = True
                position.iloc[i] = 1
            else:
                position.iloc[i] = 0

    # Ensure no NaNs and only 0/1 values
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
