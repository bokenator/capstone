"""
RSI Mean Reversion Strategy - Signal Generator

This module exports a single function `generate_signals` which produces a position
series (long-only) based on RSI mean reversion rules:

- RSI period = 14 (default, can be overridden via params['rsi_period'])
- Go long when RSI crosses below 30 (oversold)
- Exit long when RSI crosses above 70 (overbought)

The returned dictionary must contain the key 'ohlcv' mapping to a pandas Series
of position targets: 1 = long, 0 = flat. The index aligns with the provided
price series (data['ohlcv']['close']).

Dependencies: pandas, numpy
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a close price series.

    Uses Wilder's smoothing via exponential moving average with alpha=1/period.

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI as a pandas Series aligned with `close` index. Values before enough
        data to compute RSI are NaN.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    # Avoid division by zero; handle cases where avg_loss == 0
    rs = avg_gain / avg_loss

    # Build rsi with safeguards for divisions by zero
    rsi = pd.Series(index=close.index, dtype=float)

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    loss_zero = (avg_loss == 0) & ~both_zero
    valid = ~both_zero & ~loss_zero

    rsi[both_zero] = 50.0  # no change
    rsi[loss_zero] = 100.0  # gains but no losses
    rsi[valid] = 100.0 - (100.0 / (1.0 + rs[valid]))

    # Keep NaN for initial warmup period
    rsi = rsi.where(avg_gain.notna() | avg_loss.notna())

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean reversion strategy.

    Args:
        data: A dictionary containing at least the key 'ohlcv' mapping to a
            DataFrame with a 'close' column (pd.Series).
        params: Parameter dictionary. Supported keys:
            - 'rsi_period' (int): RSI lookback period, default 14
            - 'rsi_oversold' (float): Oversold threshold, default 30.0
            - 'rsi_overbought' (float): Overbought threshold, default 70.0

    Returns:
        A dict with key 'ohlcv' mapping to a pandas Series of position targets
        (1 for long, 0 for flat). The series index matches data['ohlcv']['close'].

    Raises:
        ValueError: If required data/columns are missing.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict with key 'ohlcv' mapping to a DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("`data` must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14)) if isinstance(params, dict) else 14
    oversold = float(params.get('rsi_oversold', 30.0)) if isinstance(params, dict) else 30.0
    overbought = float(params.get('rsi_overbought', 70.0)) if isinstance(params, dict) else 70.0

    if rsi_period < 1:
        raise ValueError('rsi_period must be >= 1')

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Build entry and exit signals based on crosses
    # Entry: RSI crosses below oversold (previous >= oversold and current < oversold)
    rsi_prev = rsi.shift(1)
    entry_signal = (rsi_prev >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (previous <= overbought and current > overbought)
    exit_signal = (rsi_prev <= overbought) & (rsi > overbought)

    # Construct position series: mark 1 at entry bars, 0 at exit bars, forward-fill
    position = pd.Series(index=close.index, dtype=float)
    position[:] = np.nan

    position[entry_signal.fillna(False)] = 1.0
    position[exit_signal.fillna(False)] = 0.0

    # Forward-fill and default to flat (0) before first entry
    position = position.ffill().fillna(0.0)

    # Ensure integer positions (0 or 1)
    # Clip values to {0,1} in case of accidental values
    position = position.clip(lower=0.0, upper=1.0).astype(int)

    return {"ohlcv": position}
