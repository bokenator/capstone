import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position series based on RSI mean reversion strategy.

    Strategy Logic:
    - Calculate RSI with given period (Wilder's smoothing via EWM)
    - Go long when RSI crosses below the oversold threshold
    - Exit when RSI crosses above the overbought threshold

    Args:
        data: dict with key "ohlcv" mapping to a DataFrame containing a 'close' column
        params: dict with keys 'rsi_period' (int), 'oversold' (float), 'overbought' (float)

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed same as input DataFrame.
    """
    # Input validation
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI using Wilder's smoothing (causal EWMA)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted mean with adjust=False for Wilder smoothing (causal)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Prevent division by zero; avg_loss == 0 => RS = inf => RSI = 100 (fully overbought)
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Detect crossings using previous value (causal)
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold, curr < oversold)
    entry = (rsi_prev >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought (prev <= overbought, curr > overbought)
    exit = (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series iteratively to ensure no double entries and causal behavior
    positions = np.zeros(len(close), dtype=np.int8)
    state = 0  # 0 = flat, 1 = long

    for i in range(len(close)):
        # Use only current and past information (entry/exit already computed causally)
        # If flat and entry signal occurs -> go long
        if state == 0:
            if bool(entry.iloc[i]):
                state = 1
        else:
            # If long and exit signal occurs -> flatten
            if bool(exit.iloc[i]):
                state = 0
        positions[i] = state

    position_series = pd.Series(positions, index=close.index, name="position")

    # Ensure only 0/1 and no NaNs (fill any unforeseen NaNs with 0)
    position_series = position_series.astype(int).fillna(0)

    return {"ohlcv": position_series}
