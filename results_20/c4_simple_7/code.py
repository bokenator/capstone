from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("`data` must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("`data['ohlcv']` DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Read and validate params with safe defaults from PARAM_SCHEMA
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30.0)
    overbought = params.get("overbought", 70.0)

    # Validate types
    try:
        rsi_period = int(rsi_period)
    except Exception:
        raise TypeError("rsi_period must be an int")

    try:
        oversold = float(oversold)
        overbought = float(overbought)
    except Exception:
        raise TypeError("oversold and overbought must be floats")

    # Validate bounds according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold must be strictly less than overbought")

    # Handle trivial case: empty series
    if close.empty:
        return {"ohlcv": pd.Series(dtype="int8", index=close.index)}

    # Compute RSI using Wilder's smoothing (EMA with alpha = 1 / period)
    # delta
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use EWM to approximate Wilder's smoothing
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Prepare RSI series and handle edge cases:
    # - If both avg_gain and avg_loss are zero -> RSI = 50 (no movement)
    # - If avg_loss == 0 -> RSI = 100
    # - Otherwise RSI = 100 - 100 / (1 + RS)
    rs = avg_gain / avg_loss
    rsi = pd.Series(index=close.index, dtype="float64")

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    loss_zero = (avg_loss == 0) & (~both_zero)
    normal = ~(both_zero | loss_zero)

    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[normal] = 100.0 - (100.0 / (1.0 + rs[normal]))

    # Set warmup NaNs for first rsi_period bars to avoid spurious signals
    if len(rsi) >= rsi_period:
        rsi.iloc[:rsi_period] = np.nan
    else:
        # If not enough data for a single RSI calculation, set all to NaN
        rsi[:] = np.nan

    # Generate crossing signals
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses BELOW oversold (previous >= oversold, current < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses ABOVE overbought (previous <= overbought, current > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Convert to cumulative position: +1 for each entry, -1 for each exit
    # Position is long when net entries - exits > 0
    net = entries.astype(int).cumsum() - exits.astype(int).cumsum()
    position = (net > 0).astype("int8")
    position.index = close.index

    return {"ohlcv": position}
