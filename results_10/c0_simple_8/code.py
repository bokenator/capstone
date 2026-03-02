import pandas as pd
import numpy as np
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: Lookback period for RSI.

    Returns:
        RSI series aligned with close. First `period` values are set to NaN to
        avoid warm-up bias.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Calculate price changes
    delta = close.diff()

    # Separate positive and negative gains
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/period)
    # Use adjust=False to get the recursive form
    gain_ewm = up.ewm(alpha=1.0 / period, adjust=False).mean()
    loss_ewm = down.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero; where loss is zero and gain > 0, RSI = 100
    rs = gain_ewm / loss_ewm
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle cases where loss_ewm == 0
    rsi = rsi.where(loss_ewm != 0, 100.0)

    # If both gain and loss are 0 (flat price), define RSI as 50
    flat_mask = (gain_ewm == 0) & (loss_ewm == 0)
    rsi = rsi.where(~flat_mask, 50.0)

    # Set warmup period values to NaN to avoid signals during initialization
    if period > 0:
        rsi.iloc[:period] = np.nan

    rsi.name = "rsi"
    return rsi


def generate_signals(data: dict, params: dict) -> Dict[str, pd.Series]:
    """Generate long-only position series based on RSI mean-reversion.

    Strategy logic:
    - Compute RSI with period (default 14)
    - Go long when RSI crosses below 30 (from >=30 to <30)
    - Exit when RSI crosses above 70 (from <=70 to >70)

    Args:
        data: Dict containing market data. Must contain key 'ohlcv' with a
              DataFrame that has a 'close' column.
        params: Dict of parameters. Supported keys:
            - 'rsi_period' or 'period': int RSI lookback (default 14)

    Returns:
        Dict with key 'ohlcv' mapped to a pd.Series of position targets (0 or 1),
        indexed the same as the input close prices.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing an 'ohlcv' DataFrame")

    if "ohlcv" not in data:
        raise KeyError("data must contain key 'ohlcv' with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Extract RSI period from params with sensible defaults
    if not isinstance(params, dict):
        params = {}
    rsi_period = int(params.get("rsi_period", params.get("period", 14)))
    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crossings
    prev_rsi = rsi.shift(1)

    # Entry: crosses below 30 (prev >= 30 and current < 30)
    entry_mask = (prev_rsi >= 30) & (rsi < 30)

    # Exit: crosses above 70 (prev <= 70 and current > 70)
    exit_mask = (prev_rsi <= 70) & (rsi > 70)

    # Replace NaN masks with False
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    for i in range(len(position)):
        if entry_mask.iat[i] and not in_position:
            in_position = True
        elif exit_mask.iat[i] and in_position:
            in_position = False
        position.iat[i] = 1 if in_position else 0

    position.name = "position"

    return {"ohlcv": position}
