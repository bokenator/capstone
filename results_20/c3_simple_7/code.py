import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data: dict containing "ohlcv" -> DataFrame with 'close' column
        params: dict with keys:
            - 'rsi_period' (int): RSI lookback period
            - 'oversold' (float): oversold threshold (e.g., 30)
            - 'overbought' (float): overbought threshold (e.g., 70)

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed
        the same as input close prices.
    """
    # Basic validation
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults and validation
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Edge case: empty series
    if close.shape[0] == 0:
        return {"ohlcv": pd.Series(dtype=float)}

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use ewm with adjust=False to emulate Wilder's smoothing
    # min_periods set to rsi_period to avoid early misleading values
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative strength
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases explicitly to avoid NaNs where possible
    rsi = rsi.copy()
    # If both avg_gain and avg_loss are zero (flat price), set RSI to neutral 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi.loc[both_zero.fillna(False)] = 50.0
    # If avg_loss == 0 and avg_gain > 0 -> RSI should be 100
    gain_pos_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    rsi.loc[gain_pos_loss_zero.fillna(False)] = 100.0
    # If avg_gain == 0 and avg_loss > 0 -> RSI should be 0
    loss_pos_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    rsi.loc[loss_pos_gain_zero.fillna(False)] = 0.0

    # Build position series: 0 = flat, 1 = long
    pos = pd.Series(0, index=close.index, dtype=float)

    current_pos = 0
    # Iterate sequentially to avoid lookahead and ensure deterministic behavior
    for i in range(len(close)):
        # If RSI is NaN at this bar, keep previous position
        if pd.isna(rsi.iloc[i]):
            pos.iloc[i] = current_pos
            continue

        rsi_val = float(rsi.iloc[i])

        # Entry: go long when RSI is below oversold threshold (mean-reversion)
        if current_pos == 0 and rsi_val < oversold:
            current_pos = 1
        # Exit: go flat when RSI is above overbought threshold
        elif current_pos == 1 and rsi_val > overbought:
            current_pos = 0

        pos.iloc[i] = current_pos

    # Ensure integer 0/1 values and no NaNs after warmup by filling any remaining NaNs with 0
    pos = pos.fillna(0).astype(int)

    return {"ohlcv": pos}
