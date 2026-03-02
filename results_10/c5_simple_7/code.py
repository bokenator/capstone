import pandas as pd
import numpy as np
from typing import Dict, Any


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
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Allow passing a DataFrame directly for convenience (keeps compatibility with tests)
    if isinstance(data, pd.DataFrame):
        data = {"ohlcv": data}

    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing key 'ohlcv' with a DataFrame")

    df = data["ohlcv"]
    if "close" not in df.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    close: pd.Series = df["close"].astype(float).copy()

    # Validate and extract parameters (use only declared params)
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Compute RSI using Wilder's smoothing (EWMA with adjust=False)
    # This implementation is causal (uses only past data) and deterministic.
    delta = close.diff()
    # Replace first NaN diff with 0 to avoid NaNs in initial computations
    delta.iloc[0] = 0.0

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / float(rsi_period)
    roll_up = gain.ewm(alpha=alpha, adjust=False).mean()
    roll_down = loss.ewm(alpha=alpha, adjust=False).mean()

    # Avoid division by zero; where roll_down is zero, set RSI to 100 (no losses)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Replace infinite values (from division by zero) and remaining NaNs
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    # Fill initial NaNs with neutral 50.0 (does not introduce lookahead)
    rsi = rsi.fillna(50.0)

    # Entry: RSI crosses below oversold (from >= oversold to < oversold)
    prev_rsi = rsi.shift(1)
    entry_signal = (rsi < oversold) & (prev_rsi >= oversold)
    # If previous RSI is NaN (start of series) and current is below oversold, treat as entry
    entry_signal = entry_signal | ((prev_rsi.isna()) & (rsi < oversold))

    # Exit: RSI crosses above overbought (from <= overbought to > overbought)
    exit_signal = (rsi > overbought) & (prev_rsi <= overbought)
    # If previous RSI is NaN (start) and current is above overbought, treat as exit only if we start long
    # (We do not start long by default, so no action required here)

    # Build position series (0 = flat, 1 = long). Iterate forward to avoid lookahead.
    position = pd.Series(0, index=close.index, dtype="int8")

    in_position = False
    for idx in close.index:
        # Entry: only when not already in position
        if entry_signal.loc[idx] and not in_position:
            in_position = True
        # Exit: only when currently in position
        elif exit_signal.loc[idx] and in_position:
            in_position = False

        position.loc[idx] = 1 if in_position else 0

    return {"ohlcv": position}
