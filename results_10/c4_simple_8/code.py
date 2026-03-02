import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
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
        raise TypeError("data must be a dict mapping slot names to pandas DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame that has a 'close' column")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameter extraction with basic validation
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Use only declared params
    if "rsi_period" not in params:
        raise ValueError("params must include 'rsi_period'")
    if "oversold" not in params:
        raise ValueError("params must include 'oversold'")
    if "overbought" not in params:
        raise ValueError("params must include 'overbought'")

    rsi_period = params["rsi_period"]
    oversold = params["oversold"]
    overbought = params["overbought"]

    # Type checks and conversions
    if not isinstance(rsi_period, (int, np.integer)):
        raise TypeError("rsi_period must be an integer")
    rsi_period = int(rsi_period)

    if not isinstance(oversold, (int, float, np.floating, np.integer)):
        raise TypeError("oversold must be a number")
    oversold = float(oversold)

    if not isinstance(overbought, (int, float, np.floating, np.integer)):
        raise TypeError("overbought must be a number")
    overbought = float(overbought)

    # Range checks according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if not (oversold < overbought):
        raise ValueError("oversold must be less than overbought")

    # Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use ewm for Wilder's smoothing; min_periods ensures warmup produces NaN
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative strength and RSI
    # Avoid division by zero: handle special cases explicitly
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss == 0
    # - If both avg_gain and avg_loss are zero -> RSI = 50 (no movement)
    # - If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    loss_zero = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~loss_zero, 100.0)

    gain_zero = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.where(~gain_zero, 0.0)

    # Clip to [0, 100]
    rsi = rsi.clip(lower=0.0, upper=100.0)

    # Determine cross signals: require previous value to be non-NaN to count as a cross
    prev_rsi = rsi.shift(1)

    cross_below = (prev_rsi >= oversold) & (rsi < oversold)
    cross_above = (prev_rsi <= overbought) & (rsi > overbought)

    # Exclude any crosses involving NaNs
    cross_below = cross_below & prev_rsi.notna() & rsi.notna()
    cross_above = cross_above & prev_rsi.notna() & rsi.notna()

    # Build position series: 1 when long, 0 when flat. Entry on cross_below, exit on cross_above.
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate sequentially to respect order of signals
    for i in range(len(position)):
        if not in_position:
            # Enter on a cross below
            if bool(cross_below.iat[i]):
                in_position = True
                position.iat[i] = 1
            else:
                position.iat[i] = 0
        else:
            # Currently in position; by default stay long
            position.iat[i] = 1
            # Exit on a cross above
            if bool(cross_above.iat[i]):
                in_position = False
                position.iat[i] = 0

    position.name = "position"

    return {"ohlcv": position}
