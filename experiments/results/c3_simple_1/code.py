# Complete implementation for RSI mean reversion signal generation
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _extract_close(data: Any) -> pd.Series:
    """Extract a close price series from various input types.

    Supports:
    - dict with key 'ohlcv' pointing to DataFrame with 'close' column
    - pandas DataFrame with 'close' column
    - pandas Series (assumed to be close prices)

    Raises:
        ValueError: if close prices cannot be found or extracted
    """
    if isinstance(data, dict):
        # Expect the run_backtest format
        if "ohlcv" in data:
            df = data["ohlcv"]
            if isinstance(df, pd.DataFrame) and "close" in df.columns:
                close = df["close"].astype(float)
                return close
            else:
                raise ValueError("data['ohlcv'] must be a DataFrame with a 'close' column")
        # Allow dict-like mapping directly to a Series
        # e.g., tests might pass {'close': series}
        if "close" in data:
            return pd.Series(data["close"]).astype(float)

    if isinstance(data, pd.DataFrame):
        if "close" in data.columns:
            return data["close"].astype(float)
        # If single-column DataFrame, assume it's close
        if data.shape[1] == 1:
            return data.iloc[:, 0].astype(float)

    if isinstance(data, pd.Series):
        return data.astype(float)

    raise ValueError("Unable to extract close prices from input data")


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    This implementation uses only past data (no lookahead) and returns a Series
    aligned with the input close prices. The first (period) values will be NaN
    until the averages stabilize.
    """
    # Calculate price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing via ewm with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss == 0, set RSI to 100 (no losses)
    rsi = rsi.fillna(0.0)
    rsi[avg_loss == 0] = 100.0
    # Where avg_gain == 0 and avg_loss > 0 set RSI to 0
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0.0

    return rsi


def generate_signals(data: Any, params: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """Generate long-only position targets using RSI mean reversion strategy.

    Strategy logic:
    - RSI period = 14
    - Enter long when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and curr > 70)

    Args:
        data: Input data. Can be a dict with 'ohlcv' DataFrame, a DataFrame with
              'close' column, or a Series of close prices.
        params: Optional params (not used but accepted for compatibility).

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of position targets (+1 long, 0 flat)
    """
    # Extract close prices
    close = _extract_close(data)

    n = len(close)
    if n == 0:
        raise ValueError("Input close price series is empty")

    # Compute RSI
    period = 14
    rsi = _rsi_wilder(close, period=period)

    # Determine entry and exit signals using cross logic without future data
    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= 30) & (rsi < 30)
    exit_cond = (prev_rsi <= 70) & (rsi > 70)

    # Ensure boolean series aligned and no NaNs (comparisons produce False when NaN involved)
    entry_cond = entry_cond.fillna(False)
    exit_cond = exit_cond.fillna(False)

    # Build position series respecting no double entries
    position = pd.Series(0, index=close.index, dtype=float)
    in_position = False

    # Iterate sequentially to avoid lookahead and ensure single position at a time
    for t in range(n):
        if not in_position:
            # Can only enter when not already in position
            if bool(entry_cond.iloc[t]):
                in_position = True
                position.iloc[t] = 1.0
            else:
                position.iloc[t] = 0.0
        else:
            # When in position, remain long unless exit signal triggers
            if bool(exit_cond.iloc[t]):
                in_position = False
                position.iloc[t] = 0.0
            else:
                position.iloc[t] = 1.0

    # Ensure no NaNs in output
    position = position.fillna(0.0)

    return {"ohlcv": position}
