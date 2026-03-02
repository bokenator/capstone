from typing import Dict
import pandas as pd
import numpy as np


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
    """
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain 'close' column")

    close = ohlcv["close"]

    # Validate params
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise TypeError(f"Invalid params: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100 (inclusive)")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold must be strictly less than overbought")

    def _compute_rsi(close_series: pd.Series, period: int) -> pd.Series:
        """Compute RSI using Wilder's smoothing (EWMA approximation).
        Returns a Series aligned with close_series.
        """
        # Convert to float copy to avoid modifying original
        close_s = close_series.astype(float).copy()
        delta = close_s.diff()

        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        # Use exponential moving average with alpha=1/period which approximates Wilder's smoothing.
        # min_periods ensures initial NaNs during warmup.
        alpha = 1.0 / float(period)
        avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Where avg_loss is zero -> RSI is 100 (no losses in window)
        # Where avg_gain is zero -> RSI is 0 (no gains in window)
        rsi = rsi.copy()
        rsi.loc[avg_loss == 0] = 100
        rsi.loc[avg_gain == 0] = 0

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Signals: cross below oversold -> entry; cross above overbought -> exit
    prev_rsi = rsi.shift(1)

    # Require both prev and current RSI to be not NaN for a valid cross
    valid_prev = prev_rsi.notna()
    valid_curr = rsi.notna()

    entry_mask = (prev_rsi >= oversold) & (rsi < oversold) & valid_prev & valid_curr
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought) & valid_prev & valid_curr

    # Convert booleans to cumulative counts to build position: long when entries_count > exits_count
    entries_cum = entry_mask.astype(int).cumsum()
    exits_cum = exit_mask.astype(int).cumsum()

    position = (entries_cum > exits_cum).astype(int)

    # Ensure positions are 0 where price or RSI is NaN (no trading on missing data)
    position = position.where(close.notna(), other=0)
    position = position.where(rsi.notna(), other=0)

    position = position.astype(int)
    position.name = "position"

    return {"ohlcv": position}
