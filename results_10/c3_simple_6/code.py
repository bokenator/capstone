import pandas as pd
import numpy as np
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with adjust=False).

    This implementation is causal (no lookahead) and uses past data only.

    Args:
        close: Series of close prices.
        period: RSI lookback period (e.g., 14).

    Returns:
        rsi: Series of RSI values aligned with close.index.
    """
    close = close.astype(float).copy()
    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm (adjust=False) is recursive and causal
    # alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 -> rs == inf -> rsi -> 100
    rsi = rsi.fillna(100.0).where(avg_loss != 0, 100.0)

    # If both avg_gain and avg_loss are zero (flat series), set RSI to 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position series based on RSI mean reversion.

    Strategy rules:
    - Compute RSI with period params['rsi_period']
    - Go long (position=1) when RSI crosses below params['oversold']
    - Exit to flat (position=0) when RSI crosses above params['overbought']

    Args:
        data: dict with key 'ohlcv' -> DataFrame containing a 'close' column
        params: dict containing 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' -> pd.Series of positions (0 or 1) aligned with input index
    """
    # Basic input validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Parameters with defaults and validation
    try:
        rsi_period = int(params.get('rsi_period', 14))
    except Exception:
        raise ValueError("params['rsi_period'] must be an integer")

    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions (use previous bar to detect crossings)
    prev_rsi = rsi.shift(1)

    entry_cond = (rsi < oversold) & (prev_rsi >= oversold)
    exit_cond = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series iteratively to avoid double entries/exits and ensure causal behaviour
    positions = pd.Series(0, index=close.index, dtype='int8')
    current_pos = 0

    # Use integer indexing for speed and reproducibility
    for i in range(len(close)):
        # If RSI is NaN at this point (e.g., very short series), keep current position
        rsi_val = rsi.iat[i]
        # np.isfinite covers NaN and inf
        if not np.isfinite(rsi_val):
            positions.iat[i] = current_pos
            continue

        if current_pos == 0 and entry_cond.iat[i]:
            current_pos = 1
        elif current_pos == 1 and exit_cond.iat[i]:
            current_pos = 0

        positions.iat[i] = current_pos

    # Ensure type and no NaNs
    positions = positions.astype(int).fillna(0)

    return {"ohlcv": positions}
