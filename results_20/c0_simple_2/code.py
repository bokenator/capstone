"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean-reversion strategy:
- RSI period = 14 (default)
- Go long when RSI crosses below 30
- Exit when RSI crosses above 70
- Long-only, single-asset

Exports:
- generate_signals(data: dict, params: dict) -> dict[str, pd.Series]

The returned dict must contain the key 'ohlcv' mapping to a pandas Series of
position targets (1 = long, 0 = flat) aligned with data['ohlcv']['close'] index.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close index. Values are NaN until enough data
        is available for the given period.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/period)
    # Use min_periods=period so that values are NaN until we have enough data
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss == 0 => RSI = 50 (no movement)
    # Use .loc to avoid chained-assignment warnings
    zero_loss = (avg_loss == 0) & (avg_gain > 0)
    no_move = (avg_loss == 0) & (avg_gain == 0)

    if zero_loss.any():
        rsi.loc[zero_loss] = 100.0
    if no_move.any():
        rsi.loc[no_move] = 50.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion.

    Args:
        data: Dictionary containing market data. Expected to have key 'ohlcv'
              mapped to a DataFrame with a 'close' column.
        params: Parameters dict. Supported keys (optional):
            - 'rsi_period' (int): RSI lookback period (default 14)
            - 'rsi_lower' (float): Oversold threshold (default 30)
            - 'rsi_upper' (float): Overbought threshold (default 70)

    Returns:
        A dictionary with key 'ohlcv' mapping to a pandas Series of position
        targets (1 = long, 0 = flat), indexed the same as data['ohlcv']['close'].

    Raises:
        ValueError: If input data is missing required keys/columns.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with an 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain key 'ohlcv' mapped to a DataFrame")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with sensible defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    rsi_lower = float(params.get('rsi_lower', 30.0)) if params is not None else 30.0
    rsi_upper = float(params.get('rsi_upper', 70.0)) if params is not None else 70.0

    if rsi_period <= 0:
        raise ValueError('rsi_period must be a positive integer')

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Define crossing conditions
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below rsi_lower: previous >= lower and current < lower
    entry_cond = (prev_rsi >= rsi_lower) & (rsi < rsi_lower)

    # Exit when RSI crosses above rsi_upper: previous <= upper and current > upper
    exit_cond = (prev_rsi <= rsi_upper) & (rsi > rsi_upper)

    # Build position series (0 = flat, 1 = long) by iterating through the bars.
    # This ensures we only enter when flat and only exit when long.
    position = np.zeros(len(rsi), dtype=int)
    in_long = False

    # Use .to_numpy() for faster access; keep masks as numpy bool arrays
    entry_arr = entry_cond.fillna(False).to_numpy(dtype=bool)
    exit_arr = exit_cond.fillna(False).to_numpy(dtype=bool)

    for i in range(len(rsi)):
        # If currently flat and entry condition triggered -> enter long
        if (not in_long) and entry_arr[i]:
            in_long = True
        # If currently long and exit condition triggered -> exit to flat
        elif in_long and exit_arr[i]:
            in_long = False
        position[i] = 1 if in_long else 0

    position_series = pd.Series(position, index=rsi.index, name='position')

    return {'ohlcv': position_series}
