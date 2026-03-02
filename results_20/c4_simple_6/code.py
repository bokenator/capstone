import numpy as np
import pandas as pd
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict
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

    close: pd.Series = ohlcv["close"].copy()

    # Validate params
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    # Extract and validate individual params with defaults from schema
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    oversold = float(params.get("oversold", 30.0))
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")

    overbought = float(params.get("overbought", 70.0))
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")

    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI using rolling simple average (reasonable and transparent)
    # delta
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use rolling mean with min_periods = rsi_period to avoid premature values
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # RSI will be NaN for the first rsi_period bars (warmup). Keep them as NaN.

    # Detect cross below oversold (entry) and cross above overbought (exit)
    prev_rsi = rsi.shift(1)

    # Define crossing logic: entry when previous RSI > oversold and current RSI <= oversold
    entry_sig = (prev_rsi > oversold) & (rsi <= oversold)
    # Exit when previous RSI < overbought and current RSI >= overbought
    exit_sig = (prev_rsi < overbought) & (rsi >= overbought)

    # Ensure NaNs are treated as no-signal
    entry_sig = entry_sig.fillna(False)
    exit_sig = exit_sig.fillna(False)

    # Build position series (long-only): 1 when in position, 0 otherwise
    positions = np.zeros(len(close), dtype=np.int8)
    in_pos = 0
    # Use integer indexing for speed; preserve index when converting back
    for i in range(len(close)):
        if in_pos == 0 and entry_sig.iat[i]:
            in_pos = 1
        elif in_pos == 1 and exit_sig.iat[i]:
            in_pos = 0
        positions[i] = in_pos

    pos_series = pd.Series(positions, index=close.index, name="position").astype(int)

    return {"ohlcv": pos_series}
