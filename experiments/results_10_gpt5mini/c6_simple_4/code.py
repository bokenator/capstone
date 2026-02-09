import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close.index.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's EMA smoothing: alpha = 1/period, adjust=False ensures recursive form
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero: where avg_loss == 0 -> rsi = 100 (pure gains), where both 0 -> 50
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly
    rsi = rsi.fillna(50.0)  # neutral for initial bars where RSI cannot be computed
    rsi[avg_loss == 0.0] = 100.0
    # If both avg_gain and avg_loss are 0, keep RSI at 50
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi[both_zero] = 50.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Strategy logic:
      - Calculate RSI with period = params['rsi_period']
      - Go long when RSI crosses below params['oversold'] (cross under)
      - Exit when RSI crosses above params['overbought'] (cross over)

    Args:
        data: A dict containing at least key 'ohlcv' mapping to a DataFrame with a 'close' column.
        params: Dict with keys 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of positions (0 = flat, 1 = long),
        aligned to the input 'close' index.
    """
    # Basic validation
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing key 'ohlcv' with a DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame) or 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError('rsi_period must be >= 1')

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    n = len(close)
    positions = np.zeros(n, dtype=int)

    # Iterate through time and set position based only on current and previous RSI (no lookahead)
    long = False  # current position state
    for i in range(1, n):
        prev_r = rsi.iat[i - 1]
        curr_r = rsi.iat[i]

        # Entry: when flat and RSI crosses below oversold (prev >= oversold and curr < oversold)
        if not long:
            if (not np.isnan(prev_r)) and (not np.isnan(curr_r)) and (prev_r >= oversold) and (curr_r < oversold):
                long = True
        else:
            # Exit: when long and RSI crosses above overbought (prev <= overbought and curr > overbought)
            if (not np.isnan(prev_r)) and (not np.isnan(curr_r)) and (prev_r <= overbought) and (curr_r > overbought):
                long = False

        positions[i] = 1 if long else 0

    # Ensure first value is explicit (we start flat)
    if n > 0:
        positions[0] = 0

    position_series = pd.Series(positions, index=close.index, name='position')

    return {'ohlcv': position_series}
