from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy logic:
    - Calculate RSI using Wilder smoothing approximation (EWMA with alpha=1/period)
    - Go long when RSI crosses below `oversold` (e.g., 30)
    - Exit long when RSI crosses above `overbought` (e.g., 70)

    Args:
        data: Dictionary with key "ohlcv" mapping to a DataFrame containing a 'close' column.
        params: Dictionary containing 'rsi_period' (int), 'oversold' (float), 'overbought' (float).

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 = flat, 1 = long).

    Notes:
        - The function is implemented to avoid lookahead: all computations use only data up to time t
          when producing the signal at time t.
        - Missing/NaN RSI values in the warmup period do not produce NaN positions; positions default
          to 0 (flat) until an entry condition is met.
    """

    # Basic validation
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("`data` must be a dict containing an 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("`data['ohlcv']` must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("`ohlcv` DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    n = len(close)
    # Prepare output positions (0 = flat, 1 = long)
    positions = pd.Series(0, index=close.index, dtype=int)

    if n == 0:
        return {"ohlcv": positions}

    # Compute RSI (Wilder's smoothing approximation using EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = -delta.clip(upper=0).fillna(0.0)

    # EWMA smoothing; using adjust=False gives the recursive form similar to Wilder's smoothing
    alpha = 1.0 / float(max(1, rsi_period))
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # To mimic the usual RSI warmup, set the first `rsi_period` values to NaN
    if rsi_period > 1:
        avg_gain.iloc[:rsi_period] = np.nan
        avg_loss.iloc[:rsi_period] = np.nan

    # Relative Strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle division by zero: when avg_loss == 0
    # If avg_gain == 0 too, set RSI to 50 (neutral). If avg_gain > 0, RSI -> 100.
    mask_loss_zero = (avg_loss == 0)
    if mask_loss_zero.any():
        rsi = rsi.where(~mask_loss_zero, other=np.where(avg_gain == 0, 50.0, 100.0))

    # Ensure rsi is float Series aligned with close
    rsi = pd.Series(rsi, index=close.index, dtype=float)

    # Build positions iteratively to avoid lookahead and to ensure clean entry/exit behavior
    prev_pos = 0
    for i in range(n):
        cur_rsi = rsi.iloc[i]

        # If RSI is NaN (warmup), keep previous position (default is 0)
        if np.isnan(cur_rsi):
            positions.iloc[i] = prev_pos
            continue

        prev_rsi = rsi.iloc[i - 1] if i > 0 else np.nan

        if prev_pos == 0:
            # Entry: RSI crosses below oversold.
            # Treat a NaN previous RSI as eligible for entry if current RSI is below oversold
            prev_condition = np.isnan(prev_rsi) or (prev_rsi >= oversold)
            if prev_condition and (cur_rsi < oversold):
                prev_pos = 1
        else:
            # Exit: RSI crosses above overbought.
            prev_condition = np.isnan(prev_rsi) or (prev_rsi <= overbought)
            if prev_condition and (cur_rsi > overbought):
                prev_pos = 0

        positions.iloc[i] = prev_pos

    return {"ohlcv": positions}
