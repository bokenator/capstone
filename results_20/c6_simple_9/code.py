import numpy as np
import pandas as pd
from typing import Any, Dict


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals (0 or 1) based on RSI mean reversion.

    Strategy:
    - Calculate RSI with period = params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key "ohlcv" containing a DataFrame with 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key "ohlcv" mapping to a pandas Series of positions (0 or 1)
    """
    # Validate input
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame with a 'close' column")

    close = ohlcv["close"].astype(float)

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    n = len(close)
    # Short-circuit for empty input
    if n == 0:
        return {"ohlcv": pd.Series(dtype=int)}

    # Compute RSI (Wilder's RSI using EWM with alpha=1/period)
    # This implementation uses only past data (no lookahead).
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average with alpha=1/period (Wilder smoothing)
    # min_periods=rsi_period ensures initial values are NaN until enough data
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Calculate RSI
    # rs = avg_gain / avg_loss; rsi = 100 - 100/(1+rs)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 => RSI = 0
    # - If both are zero, keep NaN (no strong signal yet)
    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.copy()
    rsi[mask_loss_zero] = 100.0
    rsi[mask_gain_zero] = 0.0

    # Determine crossing signals using previous bar (no lookahead)
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean arrays (NaNs -> False)
    entry_signals = entry_signals.fillna(False).astype(bool)
    exit_signals = exit_signals.fillna(False).astype(bool)

    # Build position series: 0 or 1, single-asset long-only, no double entries
    positions = np.zeros(n, dtype=np.int8)
    in_position = False

    for i in range(n):
        if not in_position:
            if entry_signals.iat[i]:
                in_position = True
                positions[i] = 1
            else:
                positions[i] = 0
        else:
            if exit_signals.iat[i]:
                in_position = False
                positions[i] = 0
            else:
                positions[i] = 1

    position_series = pd.Series(positions, index=close.index, dtype=int)

    return {"ohlcv": position_series}
