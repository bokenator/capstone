import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    This implementation is causal (no lookahead) by using ewm(adjust=False).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series (0-100) aligned with the input index. May contain NaN in the very
        first bars until enough history is available.
    """
    close = close.astype(float)
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: EMA with alpha = 1/period and adjust=False for a recursive, causal calculation
    # ewm with adjust=False does not use future data and is therefore safe w.r.t. lookahead bias
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Compute RSI
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases explicitly:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 => RSI = 0
    # - If both are zero (flat price) => RSI = 50 (neutral)
    rsi = rsi.astype(float)

    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)

    rsi[mask_loss_zero] = 100.0
    rsi[mask_gain_zero] = 0.0
    rsi[mask_both_zero] = 50.0

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position signals based on RSI mean-reversion.

    Strategy Logic
    - Calculate RSI with period = params['rsi_period']
    - Go long when RSI crosses below params['oversold'] (entry)
    - Exit when RSI crosses above params['overbought'] (exit)
    - Long-only, single asset

    Args:
        data: Dictionary containing 'ohlcv' DataFrame with a 'close' column.
        params: Dictionary with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' containing a pd.Series of positions (0 or 1) aligned to the input index.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with key 'ohlcv'")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Extract parameters with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError('rsi_period must be >= 1')

    # Compute RSI (causal)
    rsi = _compute_rsi(close, period=rsi_period)

    # Compute crossing signals using previous RSI (shifted) to avoid lookahead
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean series aligned and filled with False for NaNs
    entry = entry_signals.fillna(False).astype(bool).to_numpy()
    exit = exit_signals.fillna(False).astype(bool).to_numpy()

    n = len(close)
    positions = np.zeros(n, dtype=int)

    # Iterate once to build a causal position series (no double entries, no lookahead)
    # positions[i] depends only on positions[i-1] and signals at time i
    if n > 0:
        positions[0] = 0  # start flat
    for i in range(1, n):
        if positions[i-1] == 0:
            # Can only enter if currently flat and entry signal occurs
            positions[i] = 1 if entry[i] else 0
        else:
            # Currently long: stay long unless exit signal occurs
            positions[i] = 0 if exit[i] else 1

    # Ensure the result is a pandas Series aligned with input index
    position_series = pd.Series(positions, index=close.index, name='position')

    return {'ohlcv': position_series}
