import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with a DataFrame having a 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. Long-only: 1 (long) or 0 (flat).
    """

    # Extract DataFrame from input (allow passing a DataFrame directly or dict with 'ohlcv')
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        if "ohlcv" in data:
            df = data["ohlcv"]
        elif len(data) == 1:
            df = next(iter(data.values()))
        else:
            raise ValueError("Input dict must contain 'ohlcv' key mapping to a DataFrame.")
    else:
        raise TypeError("`data` must be a DataFrame or a dict containing an 'ohlcv' DataFrame")

    # Ensure 'close' column exists
    if isinstance(df, pd.Series):
        close = pd.to_numeric(df, errors="coerce")
    else:
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column")
        close = pd.to_numeric(df["close"], errors="coerce")

    # Validate params and apply defaults
    rsi_period = int(params.get("rsi_period", 14))
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100 inclusive")

    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI using Wilder's smoothing via EWM (causal; uses only past data)
    delta = close.diff()

    gains = delta.clip(lower=0.0).fillna(0.0)
    losses = (-delta.clip(upper=0.0)).fillna(0.0)

    alpha = 1.0 / float(rsi_period)
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle special cases to avoid NaNs or infs
    mask_both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi.loc[mask_both_zero] = 50.0

    mask_loss_zero = (avg_loss == 0.0) & (~mask_both_zero)
    rsi.loc[mask_loss_zero] = 100.0

    mask_gain_zero = (avg_gain == 0.0) & (~mask_both_zero)
    rsi.loc[mask_gain_zero] = 0.0

    # Replace any remaining NaNs (e.g., very first bar) with neutral 50
    rsi = rsi.fillna(50.0)

    # Crossover signals: entry when RSI crosses BELOW oversold, exit when crosses ABOVE overbought
    prev_rsi = rsi.shift(1)
    entry_signal = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signal = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series (stateful) to prevent double entries
    position = pd.Series(0, index=close.index, dtype=int)
    in_position = 0

    for i in range(len(close)):
        if bool(entry_signal.iloc[i]) and in_position == 0:
            in_position = 1
        elif bool(exit_signal.iloc[i]) and in_position == 1:
            in_position = 0

        position.iloc[i] = in_position

    return {"ohlcv": position}
