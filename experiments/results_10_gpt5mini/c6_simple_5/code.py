import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on RSI mean reversion.

    Logic:
    - Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False)
    - Go long when RSI crosses below the oversold threshold (e.g., 30)
    - Exit when RSI crosses above the overbought threshold (e.g., 70)

    Args:
        data: dict with key "ohlcv" mapping to a DataFrame that contains a 'close' column
        params: dict containing keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1)
    """

    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Helper: compute RSI using Wilder's smoothing (RMA via ewm with alpha=1/period)
    def _compute_rsi(close_s: pd.Series, period: int) -> pd.Series:
        delta = close_s.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)

        # Wilder's smoothing: exponential with alpha=1/period and adjust=False
        ma_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

        # Avoid division by zero
        rs = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle cases where both ma_up and ma_down are zero -> no movement -> RSI = 50
        mask_both_zero = (ma_up == 0) & (ma_down == 0)
        rsi = rsi.where(~mask_both_zero, 50.0)

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Signals: crossing logic (look for crosses between previous and current bar)
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series (0 or 1) by iterating forward to avoid double entries/exits
    pos = pd.Series(0, index=close.index, dtype=int)
    long = False

    # Use positional access for speed and to ensure determinism
    entry_vals = entry_signals.fillna(False).values
    exit_vals = exit_signals.fillna(False).values

    for i in range(len(pos)):
        if not long:
            if entry_vals[i]:
                long = True
                pos.iloc[i] = 1
            else:
                pos.iloc[i] = 0
        else:
            # currently long
            if exit_vals[i]:
                long = False
                pos.iloc[i] = 0
            else:
                pos.iloc[i] = 1

    # Ensure no NaN in output (positions are integers 0/1)
    pos = pos.astype(int)

    return {"ohlcv": pos}
