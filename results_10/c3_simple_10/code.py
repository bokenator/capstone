# RSI mean reversion signal generator
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (via ewm with alpha=1/period).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        RSI lookback period.

    Returns
    -------
    pd.Series
        RSI values aligned with close.index. NaN values appear for the
        initial warmup period (before enough data).
    """
    close = close.astype(float)

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing: exponential moving average with alpha=1/period
    # min_periods=period ensures NaN until we have enough data
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data (dict): data["ohlcv"] must be a DataFrame with a 'close' column.
        params (dict): must contain 'rsi_period' (int), 'oversold' (float),
                       'overbought' (float).

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series
              with values 0 (flat) or 1 (long), indexed like input close prices.
    """
    # Input validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].copy()

    # Params with defaults and validation
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        rsi_period = 14
    try:
        oversold = float(params.get("oversold", 30.0))
    except Exception:
        oversold = 30.0
    try:
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        overbought = 70.0

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Define entry/exit signals based only on past/current values (no lookahead)
    # Entry: RSI crosses below oversold -> previous >= oversold and current < oversold
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series align
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series (0 or 1) using a simple state machine to avoid double entries
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    # Iterate in order - this is deterministic and does not use future data
    for idx in range(len(close)):
        if entry_signals.iloc[idx] and not in_position:
            in_position = True
        elif exit_signals.iloc[idx] and in_position:
            in_position = False
        position.iloc[idx] = 1 if in_position else 0

    # Ensure no NaNs (positions should be explicit 0 or 1)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
