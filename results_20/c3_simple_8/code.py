import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's RSI (exponential smoothing) for a price series.

    This implementation is causal (no lookahead): each RSI value at time t
    depends only on close[:t+1]. It uses the ewm (alpha=1/period, adjust=False)
    which corresponds to Wilder's smoothing.

    Args:
        close: Series of close prices.
        period: RSI period (default 14).

    Returns:
        RSI series (float) with the same index as `close`.
    """
    close = close.astype(float)
    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via EWM with alpha=1/period
    # min_periods=period ensures we don't produce RSI until enough data is available
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals (0 or 1) based on RSI mean reversion.

    Strategy:
    - Compute RSI with given period
    - Go long when RSI crosses below `oversold` (e.g., 30)
    - Exit (go flat) when RSI crosses above `overbought` (e.g., 70)

    Args:
        data: dict containing key "ohlcv" mapping to a pd.DataFrame with a 'close' column.
        params: dict containing 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1) indexed like input.
    """
    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' mapping to a DataFrame")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI (causal)
    rsi = compute_rsi(close, period=rsi_period)

    # Signals: crossing conditions
    prev_rsi = rsi.shift(1)
    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs with False so they don't trigger signals
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series iteratively to ensure no double entries and correct statefulness
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_position = 0

    for i in range(len(close)):
        if in_position == 0 and bool(entry_signal.iat[i]):
            in_position = 1
        elif in_position == 1 and bool(exit_signal.iat[i]):
            in_position = 0
        pos.iat[i] = in_position

    return {"ohlcv": pos}
