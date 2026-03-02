from typing import Any, Dict
import pandas as pd
import numpy as np


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's RSI for a price series.

    Uses exponential smoothing with alpha = 1/period which corresponds to Wilder's
    moving average when adjust=False.

    Args:
        series: Price series (pd.Series)
        period: RSI lookback period

    Returns:
        pd.Series containing RSI values (same index as input)
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    price = series.astype(float).copy()
    delta = price.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via exponential moving average with alpha=1/period
    # Using adjust=False makes it equivalent to the recursive Wilder's formula
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is zero, RSI is 100 (no losses in window)
    rsi = rsi.fillna(0.0)
    rsi[avg_loss == 0] = 100.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate signals for an RSI mean-reversion strategy.

    Strategy rules:
    - RSI period = 14
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dictionary expected to contain key 'ohlcv' with a DataFrame that has
              a 'close' column (or a Series representing prices).
        params: Parameters dictionary (kept for compatibility; this strategy uses fixed RSI=14)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (1 for long, 0 for flat).
    """
    # Validate input
    if data is None or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing key 'ohlcv'")

    ohlcv = data['ohlcv']

    # Support both a DataFrame with 'close' column or a Series provided directly
    if isinstance(ohlcv, pd.Series):
        close = ohlcv
    else:
        if not hasattr(ohlcv, 'columns') or 'close' not in ohlcv.columns:
            raise ValueError("'ohlcv' must be a DataFrame containing a 'close' column or a Series")
        close = ohlcv['close']

    # If close is a single-column DataFrame, extract the column
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            # If multiple assets are provided, only support single-asset strategies here
            raise ValueError("generate_signals supports a single asset (a pandas Series) for 'close')")

    if not isinstance(close, pd.Series):
        raise ValueError("'close' must be a pandas Series")

    # Ensure index is sorted by time
    try:
        close = close.sort_index()
    except Exception:
        pass

    # Strategy parameters (fixed as per specification)
    rsi_period = 14
    rsi_oversold = 30.0
    rsi_overbought = 70.0

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Define entry and exit signals using crossings
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= rsi_oversold) & (rsi < rsi_oversold)
    exit_signals = (prev_rsi <= rsi_overbought) & (rsi > rsi_overbought)

    # Treat NaNs as no-signal
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series: 1 for long, 0 for flat
    index = close.index
    positions = pd.Series(0, index=index, dtype='int8')

    in_position = False
    # Iterate through bars to ensure entries/exits only applied when appropriate
    for i in range(len(index)):
        if not in_position and entry_signals.iloc[i]:
            # Enter long at this bar
            in_position = True
            positions.iloc[i] = 1
        elif in_position:
            # If in position, check for exit first
            if exit_signals.iloc[i]:
                # Exit at this bar
                in_position = False
                positions.iloc[i] = 0
            else:
                positions.iloc[i] = 1
        else:
            positions.iloc[i] = 0

    return {"ohlcv": positions}
