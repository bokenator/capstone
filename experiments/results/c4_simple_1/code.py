import typing
import pandas as pd
import numpy as np
import vectorbt as vbt


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
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """
    # Validate input data
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]

    # Read and validate parameters (use only declared params)
    rsi_period = int(params.get("rsi_period", 14))
    # enforce bounds from PARAM_SCHEMA
    if rsi_period < 2:
        rsi_period = 2
    if rsi_period > 100:
        rsi_period = 100

    oversold = float(params.get("oversold", 30.0))
    if oversold < 0.0:
        oversold = 0.0
    if oversold > 50.0:
        oversold = 50.0

    overbought = float(params.get("overbought", 70.0))
    if overbought < 50.0:
        overbought = 50.0
    if overbought > 100.0:
        overbought = 100.0

    # Compute RSI using vectorbt's RSI indicator
    # Use fully-qualified API as required
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Previous RSI (shifted by 1). Use module-qualified pd.Series.shift
    rsi_prev = pd.Series.shift(rsi, 1)

    # Entry: RSI crosses below oversold (prev >= oversold and curr < oversold)
    entries_mask = (pd.Series.fillna(rsi_prev, np.nan) >= oversold) & (rsi < oversold)
    # Exit: RSI crosses above overbought (prev <= overbought and curr > overbought)
    exits_mask = (pd.Series.fillna(rsi_prev, np.nan) <= overbought) & (rsi > overbought)

    # Clean masks: replace NaN with False to avoid spurious signals during warmup
    entries = pd.Series.fillna(entries_mask, False)
    exits = pd.Series.fillna(exits_mask, False)

    # Convert masks to numpy arrays for fast iteration
    entries_vals = entries.values.astype(bool)
    exits_vals = exits.values.astype(bool)

    n = len(close)
    pos = np.zeros(n, dtype=np.int8)  # 0 = flat, 1 = long

    # Iterate and build position vector: hold until an exit signal
    for i in range(n):
        if entries_vals[i]:
            pos[i] = 1
        else:
            if i > 0 and pos[i - 1] == 1:
                # Previously long: remain long unless exit signal
                if exits_vals[i]:
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                pos[i] = 0

    # Build pandas Series with same index as input close prices
    position_series = pd.Series(pos, index=close.index, dtype=np.int8)

    return {"ohlcv": position_series}
