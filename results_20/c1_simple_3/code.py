import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")
    ohlcv_df = data['ohlcv']
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv_df.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv_df['close'].astype(float).copy()

    # Extract and validate parameters (only use allowed params)
    # Default values are taken from PARAM_SCHEMA
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100 (inclusive)")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold level must be less than overbought level")

    # Calculate RSI using Wilder's smoothing (EMA with alpha=1/period)
    # Handle NaNs gracefully - RSI will be NaN until enough data is available
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing via ewm with alpha=1/period (adjust=False)
    alpha = 1.0 / float(rsi_period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Calculate RS and RSI, handle edge cases where avg_loss == 0 or both are zero
    # RS may produce inf when avg_loss == 0; handle those explicitly
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Create masks for special cases
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    mask_loss_zero = (avg_loss == 0) & (~mask_both_zero)
    mask_gain_zero = (avg_gain == 0) & (~mask_both_zero)

    # Ensure rsi is a Series aligned with close.index
    rsi = pd.Series(rsi, index=close.index, name='rsi').astype(float)

    # Assign values for special cases
    if mask_loss_zero.any():
        rsi.loc[mask_loss_zero] = 100.0
    if mask_gain_zero.any():
        rsi.loc[mask_gain_zero] = 0.0
    if mask_both_zero.any():
        # No price movement -> neutral
        rsi.loc[mask_both_zero] = 50.0

    # Generate entry/exit signals based on RSI crosses
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev >= oversold AND curr < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (prev <= overbought AND curr > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series: set 1 at entries, 0 at exits, forward-fill, then fill remaining with 0
    position = pd.Series(index=close.index, dtype=float)
    if entries.any():
        position.loc[entries] = 1.0
    if exits.any():
        position.loc[exits] = 0.0

    position = position.ffill().fillna(0.0).astype(int)
    position.name = 'position'

    return {'ohlcv': position}
