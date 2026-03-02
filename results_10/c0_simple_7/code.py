from typing import Dict, Any

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (exponential).

    Args:
        series: Price series (typically close prices).
        period: Lookback period for RSI (default 14).

    Returns:
        RSI as a pd.Series aligned with the input series index. The first `period`
        values are set to NaN to avoid warmup lookahead.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")

    s = series.astype(float).copy()
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via exponential moving average with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Enforce warmup NaNs for first `period` values
    rsi.iloc[:period] = np.nan
    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate signals for an RSI mean-reversion strategy.

    Strategy rules:
    - RSI period = 14 (can be overridden by params['rsi_period'])
    - Go long when RSI crosses below 30 (oversold)
    - Exit long when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dict containing at least 'ohlcv' -> pd.DataFrame with a 'close' column.
        params: Dict of parameters (optional). Recognized key: 'rsi_period'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets:
        1 for long, 0 for flat. Index matches data['ohlcv'].index.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing an 'ohlcv' DataFrame")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Parameters
    rsi_period = 14
    if isinstance(params, dict) and params.get("rsi_period") is not None:
        try:
            rsi_period = int(params.get("rsi_period"))
        except Exception:
            raise ValueError("params['rsi_period'] must be an integer")
    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI
    rsi = compute_rsi(close, period=rsi_period)

    # Define thresholds
    entry_thresh = 30.0
    exit_thresh = 70.0

    # Detect crossings: entry when RSI crosses below 30, exit when RSI crosses above 70
    rsi_prev = rsi.shift(1)
    entry_signals = ((rsi_prev >= entry_thresh) & (rsi < entry_thresh)).fillna(False)
    exit_signals = ((rsi_prev <= exit_thresh) & (rsi > exit_thresh)).fillna(False)

    # Build position series iteratively (long-only)
    position = pd.Series(0, index=close.index, dtype=int)
    prev_pos = 0
    for i in range(len(position)):
        if entry_signals.iat[i] and prev_pos == 0:
            curr_pos = 1
        elif exit_signals.iat[i] and prev_pos == 1:
            curr_pos = 0
        else:
            curr_pos = prev_pos

        position.iat[i] = curr_pos
        prev_pos = curr_pos

    return {"ohlcv": position}
