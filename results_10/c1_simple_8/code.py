import pandas as pd
import numpy as np
from typing import Dict, Any


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
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close: pd.Series = ohlcv["close"].astype(float)

    # Validate params and only use allowed keys
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    # Extract parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate parameter ranges (as per PARAM_SCHEMA)
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if not oversold < overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Handle empty series
    if close.shape[0] == 0:
        return {"ohlcv": pd.Series(dtype=float)}

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    # This implementation respects warmup by producing NaN for initial bars.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use ewm with min_periods=rsi_period to keep initial warmup NaNs
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is zero (no losses), RSI should be 100 (fully overbought)
    rsi = rsi.where(~(avg_loss == 0), 100.0)

    # Keep NaNs in warmup period to avoid false signals

    # Detect crosses: entry when RSI crosses BELOW oversold, exit when RSI crosses ABOVE overbought
    prev_rsi = rsi.shift(1)

    entry_mask = prev_rsi.notna() & rsi.notna() & (prev_rsi >= oversold) & (rsi < oversold)
    exit_mask = prev_rsi.notna() & rsi.notna() & (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series (stateful, long-only)
    pos_values = np.zeros(len(close), dtype=int)
    in_position = False

    # Iterate chronologically
    for i, (entry, exit) in enumerate(zip(entry_mask.values, exit_mask.values)):
        if not in_position and entry:
            in_position = True
        elif in_position and exit:
            in_position = False
        pos_values[i] = 1 if in_position else 0

    position = pd.Series(pos_values, index=close.index, name="position")

    return {"ohlcv": position}
