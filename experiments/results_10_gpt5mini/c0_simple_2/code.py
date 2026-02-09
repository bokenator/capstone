"""
RSI Mean Reversion Signal Generator

This module provides a single function `generate_signals` which computes RSI (14)
and returns a position series for a long-only mean reversion strategy:
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)

The returned dictionary contains the key 'ohlcv' mapping to a pd.Series of
position targets: 1 for long, 0 for flat.

The function signature matches the backtest runner's expectation and uses type
hints for all parameters and return values.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing (EMA).

    Args:
        close: Series of closing prices.
        period: RSI lookback period.

    Returns:
        RSI values as a pandas Series (float), aligned with `close` index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    close = close.astype(float)

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use Wilder's smoothing (exponential with alpha=1/period)
    # min_periods=period ensures we don't produce RSI too early
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Prepare final series and handle edge cases
    rsi = pd.Series(rsi, index=close.index, dtype=float)

    # Where both up and down are zero (no price movement), set RSI to 50 (neutral)
    mask_up_zero = roll_up == 0
    mask_down_zero = roll_down == 0

    neutral_mask = mask_up_zero & mask_down_zero
    only_down_zero = mask_down_zero & ~mask_up_zero
    only_up_zero = mask_up_zero & ~mask_down_zero

    rsi[neutral_mask] = 50.0
    rsi[only_down_zero] = 100.0  # no losses -> RSI = 100
    rsi[only_up_zero] = 0.0      # no gains -> RSI = 0

    # Fill any remaining NaNs (warmup) with NaN to avoid generating signals
    rsi = rsi.where(~rsi.isna(), np.nan)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets for an RSI mean reversion strategy.

    Args:
        data: Dictionary containing at least the key 'ohlcv' with a DataFrame
              that includes a 'close' column.
        params: Parameters dictionary. Supported keys (optional):
            - 'rsi_period' (int): RSI period (default 14)
            - 'rsi_lower' (float): Oversold threshold to enter (default 30)
            - 'rsi_upper' (float): Overbought threshold to exit (default 70)

    Returns:
        A dictionary with key 'ohlcv' mapping to a pandas Series of position
        targets: 1 for long, 0 for flat.

    Notes:
        - The strategy is long-only.
        - Entries occur when RSI crosses below the oversold threshold.
        - Exits occur when RSI crosses above the overbought threshold.
        - No signals are generated while RSI is NaN (warmup).
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict with key 'ohlcv' mapping to a DataFrame")

    if 'ohlcv' not in data:
        raise KeyError("`data` must contain key 'ohlcv'")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame")

    if 'close' not in ohlcv:
        raise KeyError("`data['ohlcv']` must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    rsi_lower = float(params.get('rsi_lower', 30.0))
    rsi_upper = float(params.get('rsi_upper', 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crosses: we require previous value to be defined (not NaN)
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below the lower threshold: prev >= lower and current < lower
    entries = (prev_rsi >= rsi_lower) & (rsi < rsi_lower) & (~rsi.isna()) & (~prev_rsi.isna())

    # Exit when RSI crosses above the upper threshold: prev <= upper and current > upper
    exits = (prev_rsi <= rsi_upper) & (rsi > rsi_upper) & (~rsi.isna()) & (~prev_rsi.isna())

    # Build position series (1 = long, 0 = flat)
    positions_arr = np.zeros(len(close), dtype=int)
    in_position = 0

    entries_arr = entries.to_numpy(dtype=bool)
    exits_arr = exits.to_numpy(dtype=bool)

    for i in range(len(positions_arr)):
        if entries_arr[i] and not in_position:
            in_position = 1
        elif exits_arr[i] and in_position:
            in_position = 0
        positions_arr[i] = in_position

    positions = pd.Series(positions_arr, index=close.index, name='position')

    return {'ohlcv': positions}
