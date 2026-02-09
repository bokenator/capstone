import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) using simple moving averages (SMA) of gains and losses.

    This implementation uses a rolling window SMA with min_periods=period, which only
    depends on current and past data (no lookahead).
    """
    close = close.astype("float64")
    delta = close.diff()

    # Gains and losses
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Average gain / loss using simple moving average (no lookahead)
    avg_gain = up.rolling(window=period, min_periods=period).mean()
    avg_loss = down.rolling(window=period, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI formula
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases:
    # - avg_loss == 0 -> RSI = 100 (no losses)
    # - avg_gain == 0 -> RSI = 0 (no gains)
    # - both 0 -> RSI = 50 (no price change)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    only_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    only_loss_zero = (avg_loss == 0) & (avg_gain > 0)

    rsi = rsi.copy()
    rsi[only_loss_zero] = 100.0
    rsi[only_gain_zero] = 0.0
    rsi[both_zero] = 50.0

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
    """
    # Accept both dict input and direct DataFrame input for convenience
    if not isinstance(data, dict):
        # Assume the passed object is the 'ohlcv' DataFrame
        data = {"ohlcv": data}

    if "ohlcv" not in data:
        raise ValueError("Input data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    ohlcv = data["ohlcv"]

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    close = ohlcv["close"].copy()

    # Extract and validate parameters (only use keys declared in PARAM_SCHEMA)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation / clamping to allowed ranges
    if rsi_period < 2:
        rsi_period = 2
    if rsi_period > 100:
        rsi_period = 100

    if oversold < 0.0:
        oversold = 0.0
    if oversold > 50.0:
        oversold = 50.0

    if overbought < 50.0:
        overbought = 50.0
    if overbought > 100.0:
        overbought = 100.0

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Prepare signals: cross below oversold -> entry, cross above overbought -> exit
    prev_rsi = rsi.shift(1)

    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    n = len(close)
    pos = np.zeros(n, dtype="int8")

    in_position = False

    # Iterate sequentially to avoid lookahead and to ensure clean state transitions
    for i in range(n):
        # If RSI is not available at this bar, remain flat until RSI is computable
        if not pd.notna(rsi.iloc[i]):
            pos[i] = 0
            continue

        if not in_position:
            # Check for entry (requires previous RSI to be available)
            if i > 0 and pd.notna(prev_rsi.iloc[i]) and entry_cond.iloc[i]:
                in_position = True
                pos[i] = 1
            else:
                pos[i] = 0
        else:
            # Currently long; default to staying long
            pos[i] = 1
            # Check for exit
            if i > 0 and pd.notna(prev_rsi.iloc[i]) and exit_cond.iloc[i]:
                in_position = False
                pos[i] = 0

    position_series = pd.Series(pos, index=close.index, name="position")

    return {"ohlcv": position_series}
