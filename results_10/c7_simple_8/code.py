import pandas as pd
import numpy as np
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing (EWMA).

    This implementation is causal (uses only present and past data) and returns
    NaN for the initial warmup period (min_periods=period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        RSI series aligned with `close` index.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via ewm with alpha=1/period, causal (adjust=False).
    # min_periods=period ensures initial values are NaN until enough data.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Relative Strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases where avg_loss == 0 or avg_gain == 0
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    # If both are zero -> RSI = 50 (neutral)
    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)

    rsi = rsi.copy()
    rsi.loc[mask_loss_zero] = 100.0
    rsi.loc[mask_gain_zero] = 0.0
    rsi.loc[mask_both_zero] = 50.0

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Strategy Logic
    - Calculate RSI with period = params['rsi_period']
    - Go long when RSI CROSSES BELOW params['oversold']
    - Exit when RSI CROSSES ABOVE params['overbought']
    - Long-only, single asset

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv'
              with a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].astype(float)

    # Extract and validate parameters (use defaults if missing)
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI (causal)
    rsi = _compute_rsi(close, rsi_period)

    # Signal generation: detect crossings using previous bar vs current bar
    rsi_prev = rsi.shift(1)

    entry_signal = ((rsi_prev >= oversold) & (rsi < oversold)).fillna(False)
    exit_signal = ((rsi_prev <= overbought) & (rsi > overbought)).fillna(False)

    # Build position series iteratively to ensure no double-entries and causal behavior
    position = pd.Series(0, index=close.index, dtype='int8')
    in_position = False

    for idx in close.index:
        if entry_signal.loc[idx] and not in_position:
            in_position = True
        elif exit_signal.loc[idx] and in_position:
            in_position = False
        position.loc[idx] = 1 if in_position else 0

    # Ensure dtype and no NaNs
    position = position.astype(int)

    return {"ohlcv": position}
