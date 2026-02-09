from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period (e.g., 14).

    Returns:
        RSI as a pandas Series aligned with `close` index.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential weighted mean with alpha=1/period and adjust=False to emulate Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases: when avg_loss == 0 => RSI = 100; when avg_gain == 0 => RSI = 0
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)

    # Fill any remaining NaNs (e.g., at the very beginning) with neutral value 50
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals based on RSI mean-reversion.

    Strategy:
    - Compute RSI with period params['rsi_period'] (default 14)
    - Enter long (position=1) when RSI crosses below params['oversold'] (default 30)
    - Exit (position=0) when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key 'ohlcv' -> DataFrame containing 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key 'ohlcv' -> pd.Series of positions (0 or 1) aligned with input index
    """
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"]
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI (causal: uses only past and current values)
    rsi = _compute_rsi(close, rsi_period)

    # Detect crossings using previous value (shifted) to avoid lookahead
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Ensure boolean numpy arrays for fast iteration
    entry_arr = entry_signals.to_numpy(dtype=bool)
    exit_arr = exit_signals.to_numpy(dtype=bool)

    # Build position series (stateful loop to enforce single long-only position)
    n = len(close)
    positions = np.zeros(n, dtype=int)
    current = 0
    for i in range(n):
        if entry_arr[i] and current == 0:
            current = 1
        elif exit_arr[i] and current == 1:
            current = 0
        positions[i] = current

    position_series = pd.Series(positions, index=close.index, name="position").astype(int)

    return {"ohlcv": position_series}
