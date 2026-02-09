# Complete implementation of the generate_signals function for RSI mean reversion

from typing import Dict

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict,
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Strategy:
    - Compute RSI with given period (Wilder smoothing via EWM)
    - Entry (go long) when RSI CROSSES BELOW the oversold level
      (previous RSI >= oversold and current RSI < oversold)
    - Exit (flatten) when RSI CROSSES ABOVE the overbought level
      (previous RSI <= overbought and current RSI > overbought)

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY: values 1 (long) or 0 (flat).
    """

    # Validate input data
    if "ohlcv" not in data:
        raise ValueError("Input data must contain 'ohlcv' key")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].astype(float).copy()

    # Extract and validate params (use defaults if keys missing)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Compute RSI (Wilder smoothing using EWM)
    # delta, gains, losses
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via exponential moving average with alpha = 1/period
    # min_periods=rsi_period ensures initial NaNs until enough data
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=rsi_period).mean()

    # Relative strength and RSI
    # Handle division by zero: when avg_loss == 0 -> RSI = 100; when avg_gain == 0 -> RSI = 0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 -> set RSI to 100.0 (no losses)
    rsi = rsi.where(~(avg_loss == 0), 100.0)
    # Where avg_gain == 0 -> set RSI to 0.0 (no gains)
    rsi = rsi.where(~(avg_gain == 0), 0.0)

    # Prepare crossover conditions. Use previous bar to detect crosses.
    rsi_prev = rsi.shift(1)

    # Only consider crosses when both current and previous RSI are finite numbers
    valid = rsi.notna() & rsi_prev.notna()

    entry_cond = valid & (rsi_prev >= oversold) & (rsi < oversold)
    exit_cond = valid & (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (0/1) by iterating through bars to ensure proper stateful behavior
    idx = close.index
    pos_vals = np.zeros(len(idx), dtype=np.int8)

    in_position = False
    for i in range(len(idx)):
        if in_position:
            # If in position and an exit occurs on this bar, exit immediately (set to 0)
            if exit_cond.iloc[i]:
                in_position = False
                pos_vals[i] = 0
            else:
                # Otherwise remain long
                pos_vals[i] = 1
        else:
            # If not in position and an entry occurs on this bar, enter (set to 1)
            if entry_cond.iloc[i]:
                in_position = True
                pos_vals[i] = 1
            else:
                pos_vals[i] = 0

    position = pd.Series(pos_vals, index=idx, name="position")

    return {"ohlcv": position}
