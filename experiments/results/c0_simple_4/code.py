"""
RSI Mean Reversion Signal Generator

Exports:
- generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

Strategy:
- RSI period 14 (default)
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long-only

The function is robust to NaNs and warmup periods.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close index. Values may contain NaN for warmup.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: ewm with alpha=1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    # Where avg_loss is zero -> set RSI to 100 (no losses), where avg_gain is zero -> RSI=0 (no gains)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle infinite/NaN cases explicitly
    rsi = rsi.where(~avg_loss.isna())
    rsi = rsi.fillna(50.0)

    # If avg_loss is 0 -> RSI 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    # If avg_gain is 0 -> RSI 0
    rsi = rsi.where(avg_gain != 0, 0.0)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for a single-asset RSI mean reversion strategy.

    Args:
        data: Dict containing at least data['ohlcv'] which is a DataFrame with a 'close' column.
        params: Dict of parameters. Supported keys:
            - 'rsi_period' (int): lookback period for RSI (default 14)
            - 'rsi_lower' (float): lower threshold to enter long (default 30)
            - 'rsi_upper' (float): upper threshold to exit long (default 70)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets: 1.0 for long, 0.0 for flat.

    Notes:
        - Signals are generated only after RSI warmup; until then the position is 0.
        - The function is long-only.
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict of DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    rsi_lower = float(params.get("rsi_lower", 30.0))
    rsi_upper = float(params.get("rsi_upper", 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be > 0")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Determine cross-under (entry) and cross-over (exit)
    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= rsi_lower) & (rsi < rsi_lower) & rsi.notna() & prev_rsi.notna()
    exit_cond = (prev_rsi <= rsi_upper) & (rsi > rsi_upper) & rsi.notna() & prev_rsi.notna()

    # Build position series by forward filling entry/exit markers
    position = pd.Series(index=rsi.index, dtype=float)
    position[:] = np.nan

    # Mark entries as 1.0 and exits as 0.0
    if entry_cond.any():
        position.loc[entry_cond] = 1.0
    if exit_cond.any():
        position.loc[exit_cond] = 0.0

    # Forward-fill signals to maintain position between entry and exit
    position = position.ffill().fillna(0.0)

    # Ensure same dtype and name
    position = position.astype(float)
    position.name = "position"

    return {"ohlcv": position}
