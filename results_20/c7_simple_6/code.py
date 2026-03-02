# Auto-generated RSI mean reversion signal generator
import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EWMA with alpha=1/period).

    This implementation is causal (no lookahead): it only uses past and current values
    to compute the RSI at each time step.

    Args:
        close: Price series (pd.Series) of close prices.
        period: RSI lookback period (int).

    Returns:
        pd.Series of RSI values aligned with the input index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Calculate price differences
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha = 1/period (Wilder's smoothing)
    # adjust=False makes it recursive and causal
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    # Handle division by zero: if avg_loss == 0 -> RS = inf -> RSI = 100
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is zero (no losses), set RSI to 100
    rsi = rsi.where(~(avg_loss == 0), 100.0)

    return rsi


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
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with key 'ohlcv' mapping to a DataFrame")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame that has a 'close' column")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Validate and extract params (use only allowed params)
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Basic parameter validation against schema limits
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Detect crosses: entry when RSI crosses below oversold, exit when crosses above overbought
    prev_rsi = rsi.shift(1)

    # Use >= and <= to ensure we detect crossing when equal previously
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN with False to avoid ambiguous boolean logic
    entry_mask = entry_mask.fillna(False)
    exit_mask = exit_mask.fillna(False)

    # Build position series: iterate through time to avoid double entries/exits
    positions = np.zeros(len(close), dtype=np.int8)
    in_position = 0

    for i in range(len(close)):
        if in_position == 0 and entry_mask.iat[i]:
            in_position = 1
        elif in_position == 1 and exit_mask.iat[i]:
            in_position = 0
        positions[i] = in_position

    position_series = pd.Series(positions, index=close.index, name='position')

    return {'ohlcv': position_series}
