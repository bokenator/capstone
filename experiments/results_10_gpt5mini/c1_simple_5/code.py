import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy:
    - Compute RSI (Wilder's smoothing approximation via EWM) using rsi_period
    - Go long when RSI crosses below `oversold` (cross from above to <= oversold)
    - Exit long when RSI crosses above `overbought` (cross from below to >= overbought)
    - Long-only: positions are 1 (long) or 0 (flat)

    Args:
        data: Dict with key 'ohlcv' mapping to a DataFrame with a 'close' column.
        params: Dict with keys 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        Dict mapping 'ohlcv' to a pandas Series of positions (0 or 1), indexed like the input.
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain an 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    # Extract close prices and ensure float dtype
    close = ohlcv["close"].astype(float).copy()

    # Validate params and use provided defaults if missing
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Only use allowed parameter keys
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    # Validate parameter ranges according to PARAM_SCHEMA
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Handle empty series
    if close.empty:
        return {"ohlcv": pd.Series(dtype=int, index=close.index)}

    # Compute RSI using Wilder's smoothing approximation with EWM
    # Delta
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Smoothed averages via EWM (alpha = 1 / period) approximates Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Relative strength and RSI
    # Handle division by zero gracefully: where avg_loss == 0 -> RSI = 100; where avg_gain == 0 -> RSI = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Enforce warmup: require at least rsi_period bars before signals
    if rsi_period > 0:
        rsi.iloc[:rsi_period] = np.nan

    # Prepare previous RSI for crossing detection
    rsi_prev = rsi.shift(1)

    # Entry: RSI crosses below oversold (was above oversold previously)
    entry_mask = (
        rsi_prev.notna()
        & rsi.notna()
        & (rsi_prev > oversold)
        & (rsi <= oversold)
    )

    # Exit: RSI crosses above overbought (was below overbought previously)
    exit_mask = (
        rsi_prev.notna()
        & rsi.notna()
        & (rsi_prev < overbought)
        & (rsi >= overbought)
    )

    # Build position series (stateful loop to ensure single-asset long-only behavior)
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = False

    for idx in close.index:
        if entry_mask.loc[idx] and not in_position:
            in_position = True
        elif exit_mask.loc[idx] and in_position:
            in_position = False

        position.loc[idx] = 1 if in_position else 0

    return {"ohlcv": position}
