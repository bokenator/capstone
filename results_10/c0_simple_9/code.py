"""
RSI Mean Reversion Signal Generator

This module provides the generate_signals function which computes RSI (period 14 by default)
and produces long-only position signals:

- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)

The returned dictionary contains at least the key 'ohlcv' mapping to a pandas Series
of position targets (+1 for long, 0 for flat), aligned with the provided ohlcv['close'].

Author: Auto-generated
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing (EWMA).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series (float), same index as `close`. Early values may be NaN if input has NaNs.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")

    close = close.astype(float)
    # Price changes
    delta = close.diff()

    # Gains and losses
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use Wilder's smoothing via EWM with alpha=1/period and adjust=False
    # This is the commonly-used RSI implementation.
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero; where roll_down == 0 and roll_up == 0 -> RSI undefined (set to NaN)
    rs = roll_up / roll_down

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where roll_down is zero and roll_up is positive, RSI should be 100.
    mask_down_zero = (roll_down == 0) & (roll_up > 0)
    rsi.loc[mask_down_zero] = 100.0

    # Where both roll_up and roll_down are zero, set RSI to NaN to avoid spurious signals
    mask_both_zero = (roll_down == 0) & (roll_up == 0)
    rsi.loc[mask_both_zero] = np.nan

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - Compute RSI (period=14 by default)
    - Go long when RSI crosses below 30
    - Exit when RSI crosses above 70
    - Long-only

    Args:
        data: Dictionary containing market data. Expects data['ohlcv'] to be a DataFrame with a 'close' column.
        params: Parameters dictionary. Recognized keys:
            - 'rsi_period' (int): RSI lookback period. Defaults to 14.

    Returns:
        A dict with at least the key 'ohlcv' mapping to a pd.Series of position targets (+1 long, 0 flat),
        indexed the same as data['ohlcv']['close']. Also returns the computed 'rsi' series for convenience.

    Raises:
        ValueError: If required data is missing or invalid.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")

    ohlcv = data.get("ohlcv")
    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data must contain 'ohlcv' as a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters
    rsi_period = int(params.get("rsi_period", 14)) if isinstance(params, dict) else 14
    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI
    rsi = compute_rsi(close, period=rsi_period)

    # Detect crosses: use previous bar to define a cross
    rsi_prev = rsi.shift(1)

    # Conditions: require both current and previous RSI to be non-NaN to consider a cross
    non_na_prev = rsi_prev.notna()
    non_na_curr = rsi.notna()

    # Entry: RSI crosses below 30 (prev >= 30 and curr < 30)
    entry_mask = (rsi_prev >= 30.0) & (rsi < 30.0) & non_na_prev & non_na_curr

    # Exit: RSI crosses above 70 (prev <= 70 and curr > 70)
    exit_mask = (rsi_prev <= 70.0) & (rsi > 70.0) & non_na_prev & non_na_curr

    # Build position series: NaN where no explicit command, then forward-fill and set initial to 0
    position = pd.Series(index=close.index, dtype=float)
    position.loc[:] = np.nan

    # Set entry/exit commands
    position.loc[entry_mask] = 1.0
    position.loc[exit_mask] = 0.0

    # Forward-fill commands to maintain position until next command; start flat (0)
    position = position.ffill().fillna(0.0)

    # Ensure positions are only 0 or 1
    position = position.clip(lower=0.0, upper=1.0)

    return {"ohlcv": position, "rsi": rsi}
