from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series (0-100) aligned with the input index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Work on a float copy and fill missing values to avoid propagate NaNs
    close = close.astype(float).copy()
    if close.isna().all():
        # All NaNs -> return series of NaNs
        return pd.Series(index=close.index, data=np.nan, dtype=float)

    # Forward/backward fill to handle occasional NaNs while preserving index length
    close = close.ffill().bfill()

    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's smoothing: ewm with alpha=1/period and adjust=False
    # This approximates the recursive Wilder moving average
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = roll_up / roll_down
    # Where roll_down is zero and roll_up is zero -> RSI = 50
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle cases where roll_down == 0
    rsi = rsi.where(~(roll_down == 0), other=100.0)
    # If both up and down are zero (no price movement), set RSI to 50
    both_zero = (roll_down == 0) & (roll_up == 0)
    rsi = rsi.where(~both_zero, other=50.0)

    # Keep original index and name
    rsi.name = "rsi"
    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute 14-period RSI
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single-asset

    Args:
        data: Dictionary containing at least the key 'ohlcv' with a DataFrame
              that has a 'close' column.
        params: Parameters dictionary (optional). Supported keys:
            - 'rsi_period' (int): RSI lookback period (default: 14)

    Returns:
        A dict with key 'ohlcv' mapped to a pd.Series of position targets
        where 1 = long, 0 = flat. The series is aligned with data['ohlcv'].index.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' -> DataFrame")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame with a 'close' column")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Parameters
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Entry when RSI crosses below 30: previous >= 30 and current < 30
    entry_thresh = 30.0
    exit_thresh = 70.0

    prev_rsi = rsi.shift(1)

    entries = (rsi < entry_thresh) & (prev_rsi >= entry_thresh)
    exits = (rsi > exit_thresh) & (prev_rsi <= exit_thresh)

    # Replace NaNs with False to avoid accidental signals during warmup
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build position series: 1 when in long, 0 when flat
    n = len(rsi)
    pos_arr = np.zeros(n, dtype=int)
    in_long = False

    # Use integer indexing for speed and to avoid SettingWithCopy issues
    entries_vals = entries.to_numpy(dtype=bool)
    exits_vals = exits.to_numpy(dtype=bool)

    for i in range(n):
        if entries_vals[i] and not in_long:
            in_long = True
            pos_arr[i] = 1
        elif exits_vals[i] and in_long:
            in_long = False
            pos_arr[i] = 0
        else:
            pos_arr[i] = 1 if in_long else 0

    position_series = pd.Series(pos_arr, index=close.index, name="position")

    return {"ohlcv": position_series}
