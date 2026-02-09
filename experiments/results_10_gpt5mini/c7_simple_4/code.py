from typing import Dict

import numpy as np
import pandas as pd


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - Compute RSI (Wilder) with period `rsi_period`.
    - Go long (position = 1) when RSI CROSSES BELOW `oversold` (from above to <=).
    - Exit (position = 0) when RSI CROSSES ABOVE `overbought` (from below to >=).
    - Long-only, single asset. Returns a Series of 0/1 position targets.

    Args:
        data: Dict with key 'ohlcv' mapping to a DataFrame containing a 'close' column.
        params: Dict with keys 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        Dict mapping 'ohlcv' to a pd.Series of positions (0 or 1) indexed like the input.
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv' mapping to a DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")
    df = data['ohlcv']
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in df.columns:
        raise ValueError("data['ohlcv'] DataFrame must contain a 'close' column")

    close: pd.Series = df['close'].astype('float64')

    n = len(close)

    # Handle empty input
    if n == 0:
        return {"ohlcv": pd.Series(dtype='int64')}

    # Read and validate params (only use declared params)
    try:
        rsi_period = int(params.get('rsi_period', 14))
        oversold = float(params.get('oversold', 30.0))
        overbought = float(params.get('overbought', 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if not (oversold < overbought):
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use min_periods equal to rsi_period to avoid spurious early values
    # Wilder's smoothing via EWM with alpha=1/period and adjust=False
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative Strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss == 0
    # If avg_gain == 0 and avg_loss == 0 -> RSI = 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi[both_zero] = 50.0

    loss_zero = (avg_loss == 0) & (avg_gain > 0)
    rsi[loss_zero] = 100.0

    gain_zero = (avg_gain == 0) & (avg_loss > 0)
    rsi[gain_zero] = 0.0

    # Ensure rsi is a float Series aligned with close
    rsi = rsi.reindex(close.index)

    # Signals: detect crosses using only past information (shifted previous value)
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi > oversold) & (rsi <= oversold)
    exit_signal = (prev_rsi < overbought) & (rsi >= overbought)

    # Replace NaN signals (from warmup) with False to avoid spurious triggers
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series iteratively to avoid lookahead
    position = pd.Series(0, index=close.index, dtype='int64')

    # Convert to numpy arrays for speed
    entry_arr = entry_signal.to_numpy(dtype=bool)
    exit_arr = exit_signal.to_numpy(dtype=bool)

    prev_pos = 0
    # Iterate through bars sequentially
    for i in range(n):
        if i == 0:
            pos_i = 0
        else:
            if prev_pos == 0 and entry_arr[i]:
                pos_i = 1
            elif prev_pos == 1 and exit_arr[i]:
                pos_i = 0
            else:
                pos_i = prev_pos
        position.iat[i] = pos_i
        prev_pos = pos_i

    # Ensure output contains only 0 or 1 and has no NaNs after warmup
    position = position.astype('int64')

    return {"ohlcv": position}
