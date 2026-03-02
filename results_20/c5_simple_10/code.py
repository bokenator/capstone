import pandas as pd
import numpy as np
from typing import Dict


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

    # Allow passing DataFrame directly for convenience, but prefer dict with 'ohlcv'
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if 'ohlcv' not in data:
            raise ValueError("data must be a dict containing 'ohlcv' DataFrame with 'close' column")
        ohlcv = data['ohlcv']
    else:
        raise ValueError("data must be a pandas DataFrame or a dict with 'ohlcv' DataFrame")

    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("'ohlcv' must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    # Extract parameters (use only allowed params)
    rsi_period = int(params.get('rsi_period', 14))
    # enforce bounds from PARAM_SCHEMA
    if rsi_period < 2:
        rsi_period = 2
    if rsi_period > 100:
        rsi_period = 100

    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    close = ohlcv['close'].astype(float).copy()

    # Compute RSI (Wilder's smoothing via ewm, causal - no lookahead)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha=1/period (Wilder's smoothing)
    # min_periods ensures RSI is NaN until we have enough data
    avg_up = up.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_down = down.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    # Relative Strength and RSI
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))

    # Signals: crossing events (uses only current and previous RSI values -> no lookahead)
    prev_rsi = rsi.shift(1)

    entry_signal = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signal = (rsi > overbought) & (prev_rsi <= overbought)

    # Build position series: 1 = long, 0 = flat. Iterate once to avoid double entries/exits.
    position = pd.Series(0, index=close.index, dtype='int8')

    in_position = 0
    # Use .itertuples for speed while preserving index mapping
    # But we need indexed boolean access; we can use integer-based iteration
    for i in range(len(position)):
        if in_position == 0:
            if bool(entry_signal.iloc[i]):
                in_position = 1
        else:
            if bool(exit_signal.iloc[i]):
                in_position = 0
        position.iloc[i] = in_position

    # Ensure no NaNs in position (shouldn't be any), and dtype is numeric
    position = position.astype('int8')

    return {"ohlcv": position}
