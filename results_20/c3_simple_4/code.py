"""
RSI Mean Reversion Signal Generator

Implements generate_signals(data: dict, params: dict) -> dict
- Expects data["ohlcv"] to be a DataFrame with a 'close' column
- params must contain: 'rsi_period', 'oversold', 'overbought'

Signals: long-only (0 or 1)
- Enter long when RSI crosses below oversold
- Exit when RSI crosses above overbought

This implementation uses Wilder's smoothing via pandas ewm (causal, no lookahead).
Edge cases handled: NaNs in price series (forward/back filled), small series lengths, deterministic behavior.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: price series
        period: RSI lookback period (int)

    Returns:
        pd.Series: RSI values (same index as close)
    """
    if period <= 0:
        raise ValueError("rsi period must be > 0")

    # Work on a float copy and fill missing prices using forward/back fill (causal)
    close = close.astype(float).copy()
    # Forward-fill missing prices; then backfill if leading NaNs exist
    close_filled = close.fillna(method="ffill").fillna(method="bfill")

    # Price changes
    delta = close_filled.diff()

    # Gains and losses
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)

    # Wilder's smoothing via ewm (alpha=1/period). This is causal (no lookahead).
    # adjust=False makes it recursive like Wilder's original formulation.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss

    # RSI formula
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle case when avg_gain and avg_loss are both zero -> RSI undefined => neutral 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # Keep same index
    rsi.name = "rsi"
    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position series based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" -> DataFrame containing 'close' column
              (also accepts passing the ohlcv DataFrame directly)
        params: dict with keys:
            - 'rsi_period' (int)
            - 'oversold' (float)
            - 'overbought' (float)

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series
              aligned with input close index and contains 0 (flat) or 1 (long)
    """
    # Accept either a dict or a DataFrame (some tests may pass the DataFrame directly)
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict) and "ohlcv" in data:
        ohlcv = data["ohlcv"]
    else:
        raise ValueError("data must be a dict with key 'ohlcv' or a DataFrame")

    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", params.get("rsiPeriod", 14)))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Replace any remaining NaNs in RSI with neutral 50 to avoid spurious signals
    rsi_safe = rsi.fillna(50.0)

    # Previous RSI (shifted by one). Fill first previous with current to avoid detecting
    # a spurious crossing on the very first bar.
    prev_rsi = rsi_safe.shift(1).fillna(rsi_safe)

    # Prepare position array
    n = len(close)
    positions = np.zeros(n, dtype=np.int8)

    in_position = False

    # Iterate once (causal). Using Python loop keeps behavior explicit and deterministic.
    for i in range(n):
        cur = float(rsi_safe.iloc[i])
        prev = float(prev_rsi.iloc[i])

        if not in_position:
            # Entry: RSI crosses below oversold (prev >= oversold and cur < oversold)
            if (prev >= oversold) and (cur < oversold):
                in_position = True
        else:
            # Exit: RSI crosses above overbought (prev <= overbought and cur > overbought)
            if (prev <= overbought) and (cur > overbought):
                in_position = False

        positions[i] = 1 if in_position else 0

    position_series = pd.Series(positions, index=close.index, name="position")

    return {"ohlcv": position_series}
