from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position signals (0 or 1) based on RSI mean reversion.

    Strategy rules:
    - RSI period = params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit long when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key "ohlcv" containing a DataFrame with a 'close' column
        params: dict with keys 'rsi_period', 'oversold', 'overbought'

    Returns:
        dict with key "ohlcv" mapping to a pandas Series of positions (0 or 1),
        indexed the same as input close prices.
    """
    # Validate input
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults and validation
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Prepare arrays
    close_values = close.to_numpy(dtype=float)
    n = rsi_period
    length = len(close_values)

    # Empty output for empty input
    if length == 0:
        return {"ohlcv": pd.Series(dtype=float)}

    # Compute deltas
    delta = np.empty(length, dtype=float)
    delta[0] = np.nan
    if length > 1:
        delta[1:] = close_values[1:] - close_values[:-1]

    # Gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Prepare average gain/loss arrays using Wilder's smoothing
    avg_gain = np.full(length, np.nan, dtype=float)
    avg_loss = np.full(length, np.nan, dtype=float)

    # Need at least n+1 points to compute the first average (since delta[0] is NaN)
    if length > n:
        # Initial simple average of the first n periods (using deltas indices 1..n)
        initial_gain = gains[1 : n + 1].mean()
        initial_loss = losses[1 : n + 1].mean()
        avg_gain[n] = initial_gain
        avg_loss[n] = initial_loss

        # Wilder smoothing for subsequent values
        for i in range(n + 1, length):
            avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gains[i]) / n
            avg_loss[i] = (avg_loss[i - 1] * (n - 1) + losses[i]) / n

    # Compute RSI array (NaN for indices < n)
    rsi = np.full(length, np.nan, dtype=float)
    for i in range(n, length):
        ag = avg_gain[i]
        al = avg_loss[i]
        # Handle division by zero cases
        if np.isnan(ag) and np.isnan(al):
            rsi[i] = np.nan
        else:
            if al == 0.0:
                # If there is no loss, RSI is 100 (if gains > 0) or 50 if no movement
                if ag == 0.0:
                    rsi[i] = 50.0
                else:
                    rsi[i] = 100.0
            else:
                rs = ag / al
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    rsi_series = pd.Series(rsi, index=close.index)

    # Generate position series: 0 = flat, 1 = long
    positions = np.zeros(length, dtype=int)
    state = 0  # current position: 0 flat, 1 long

    for i in range(1, length):
        prev_rsi = rsi[i - 1]
        cur_rsi = rsi[i]

        # If either RSI is NaN, keep previous state (no signal)
        if np.isnan(prev_rsi) or np.isnan(cur_rsi):
            positions[i] = state
            continue

        # Entry: RSI crosses below oversold (prev >= oversold and cur < oversold)
        if state == 0 and (prev_rsi >= oversold) and (cur_rsi < oversold):
            state = 1
        # Exit: RSI crosses above overbought (prev <= overbought and cur > overbought)
        elif state == 1 and (prev_rsi <= overbought) and (cur_rsi > overbought):
            state = 0

        positions[i] = state

    position_series = pd.Series(positions, index=close.index, dtype=int)

    return {"ohlcv": position_series}
