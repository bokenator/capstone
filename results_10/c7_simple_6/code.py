import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean-reversion strategy.

    Strategy:
    - Compute RSI with Wilder smoothing (period = params['rsi_period']).
    - Entry (go long -> position = 1) when RSI crosses BELOW `oversold`.
      Condition: previous RSI >= oversold and current RSI < oversold.
    - Exit (go flat -> position = 0) when RSI crosses ABOVE `overbought`.
      Condition: previous RSI <= overbought and current RSI > overbought.
    - Long-only. Position values are 1 (long) or 0 (flat).

    Args:
        data: Dict containing 'ohlcv' -> DataFrame with 'close' column.
              For convenience, a direct DataFrame may also be provided (treated as 'ohlcv').
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed
        the same as the input close prices.
    """
    # Accept either a dict with key 'ohlcv' or a direct DataFrame for robustness
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("data must contain 'ohlcv' key with a DataFrame")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("data must be a dict[str, DataFrame] or a DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("'ohlcv' DataFrame must contain 'close' column")

    # Extract and validate parameters (use only allowed params)
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period out of allowed range [2, 100]")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold out of allowed range [0.0, 50.0]")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought out of allowed range [50.0, 100.0]")

    close = ohlcv["close"].astype(float).copy()

    # Compute RSI using Wilder smoothing (EMA with alpha=1/period)
    # delta uses previous close => no lookahead
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use ewm with adjust=False to replicate Wilder's smoothing (no lookahead)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative strength
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle infinite/div-by-zero and initial NaNs: set neutral 50 for undefined RSI
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    # Detect crossings (use previous value via shift) and fill NaNs with False
    enters = ((rsi.shift(1) >= oversold) & (rsi < oversold)).fillna(False)
    exits = ((rsi.shift(1) <= overbought) & (rsi > overbought)).fillna(False)

    # Build position series iteratively to avoid double entries
    position = pd.Series(0, index=close.index, dtype="int8")
    in_long = False

    # Use positional access for speed and determinism
    for i in range(len(close)):
        if not in_long:
            if enters.iat[i]:
                in_long = True
                position.iat[i] = 1
            else:
                position.iat[i] = 0
        else:
            # currently in long
            if exits.iat[i]:
                in_long = False
                position.iat[i] = 0
            else:
                position.iat[i] = 1

    return {"ohlcv": position}
