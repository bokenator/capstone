from typing import Dict, Any

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
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = df["close"].astype("float64").copy()

    # Extract and validate params
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    # Standard RSI calculation (no lookahead): use past differences only
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use ewm with alpha=1/period and min_periods=period to match Wilder's RSI
    # This will produce NaN for the first (period-1) values
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Avoid division by zero: where avg_loss == 0 -> RSI = 100, where avg_gain == 0 -> RSI = 0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle cases where avg_loss is zero (rs = inf) -> RSI should be 100
    rsi = rsi.where(~np.isinf(rs), 100.0)

    # At early indices RSI will be NaN (insufficient data). That's expected.

    # Generate entry/exit signals based only on current and past RSI (no lookahead)
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN booleans with False to avoid propagation
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    n = len(close)
    positions = np.zeros(n, dtype=np.int8)

    # Build position series iteratively to ensure we don't enter when already long
    for i in range(1, n):
        if entry_signal.iat[i] and positions[i - 1] == 0:
            positions[i] = 1
        elif exit_signal.iat[i] and positions[i - 1] == 1:
            positions[i] = 0
        else:
            positions[i] = positions[i - 1]

    # Ensure first position is 0 (flat)
    positions[0] = 0

    position_series = pd.Series(positions.astype("int64"), index=close.index, name="position")

    return {"ohlcv": position_series}
