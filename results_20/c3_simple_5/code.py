import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on an RSI mean-reversion strategy.

    Strategy:
    - Compute RSI with Wilder smoothing (period = params['rsi_period']).
    - Go long (position=1) when RSI crosses below the oversold threshold.
    - Exit (position=0) when RSI crosses above the overbought threshold.

    Args:
        data: dict containing 'ohlcv' DataFrame with a 'close' column, or a DataFrame itself.
        params: dict containing keys:
            - 'rsi_period' (int): RSI lookback period
            - 'oversold' (float): oversold threshold (e.g., 30.0)
            - 'overbought' (float): overbought threshold (e.g., 70.0)

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series of 0/1 values
              (0 = flat, 1 = long) aligned with the input close index.
    """

    # Accept either a dict with 'ohlcv' or a DataFrame directly
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data dict must contain 'ohlcv' key with OHLCV DataFrame")
        df = data["ohlcv"]
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("data must be a dict with 'ohlcv' DataFrame or a DataFrame")

    if "close" not in df.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = df["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    n = len(close)
    if n == 0:
        # Return empty series if no data
        return {"ohlcv": pd.Series(dtype=float)}

    # Compute RSI (Wilder's smoothing using EWM with adjust=False -> causal)
    delta = close.diff()

    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = -delta.clip(upper=0.0).fillna(0.0)

    # Wilder's smoothing via EWM (causal). alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Relative strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle divide by zero and any NaNs by filling with extreme RSI values
    # If avg_loss == 0 -> RSI = 100, if avg_gain == 0 -> RSI = 0
    rsi = rsi.fillna(100.0)
    rsi = rsi.clip(0.0, 100.0)

    # Entry: RSI crosses below oversold (previous >= oversold and current < oversold)
    # Exit: RSI crosses above overbought (previous <= overbought and current > overbought)
    prev_rsi = rsi.shift(1)

    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs with False to avoid accidental triggers
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series with a simple state machine to avoid double-entries
    pos = np.zeros(n, dtype=int)
    state = 0  # 0 = flat, 1 = long
    for i in range(n):
        if state == 0:
            if entry_mask.iat[i]:
                state = 1
        else:  # state == 1
            if exit_mask.iat[i]:
                state = 0
        pos[i] = state

    position_series = pd.Series(pos, index=close.index, name="position")

    return {"ohlcv": position_series}
