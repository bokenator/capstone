from typing import Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI period (int).

    Returns:
        RSI series (float) with the same index as close. Initial values before
        enough data are NaN.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (EMA with alpha=1/period). Use min_periods=period so
    # the first valid RSI appears after `period` bars.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Clip to [0, 100] and preserve NaNs
    rsi = rsi.clip(lower=0.0, upper=100.0)

    return rsi


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. This is a LONG-ONLY strategy,
        so position values are: 1 (long) or 0 (flat).
    """
    # Basic validations
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot name to DataFrame")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    # Extract close prices (work on a copy to avoid modifying input)
    close = ohlcv["close"].astype(float).copy()

    # Retrieve and validate parameters (only allowed params are accessed)
    try:
        rsi_period = int(params["rsi_period"])
        oversold = float(params["oversold"])
        overbought = float(params["overbought"])
    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings. Require both previous and current RSI to be finite to
    # avoid spurious signals around NaNs/warmup periods.
    prev_rsi = rsi.shift(1)

    enter = (prev_rsi >= oversold) & (rsi < oversold) & prev_rsi.notna() & rsi.notna()
    exit = (prev_rsi <= overbought) & (rsi > overbought) & prev_rsi.notna() & rsi.notna()

    # Build position series (long-only). Iterate to ensure proper state handling.
    pos = pd.Series(0, index=close.index, dtype="int8")
    in_long = 0

    # Use iloc for position; this is efficient enough for typical backtests
    for i in range(len(close)):
        if in_long == 0 and enter.iloc[i]:
            in_long = 1
        elif in_long == 1 and exit.iloc[i]:
            in_long = 0
        pos.iloc[i] = in_long

    # Ensure no NaNs and correct dtype
    pos = pos.astype(int)

    return {"ohlcv": pos}
