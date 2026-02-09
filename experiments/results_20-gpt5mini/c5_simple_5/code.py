import numpy as np
import pandas as pd
from typing import Any, Dict


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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """
    # Validate input structure
    if isinstance(data, pd.DataFrame):
        # Allow convenience: if a DataFrame is passed directly, treat it as ohlcv
        df = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")
        df = data["ohlcv"]
    else:
        raise TypeError("data must be a dict with key 'ohlcv' or a pandas DataFrame")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in df.columns:
        raise KeyError("Input DataFrame must contain 'close' column")

    # Extract close prices
    close = df["close"].astype(float).copy()

    # Parameters with defaults (as per PARAM_SCHEMA)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation / clamping
    if rsi_period < 2:
        rsi_period = 2
    if rsi_period > 100:
        rsi_period = 100

    # Compute RSI (Wilder's smoothing via ewm with alpha=1/period)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use Wilder's smoothing (recursive) - causal, no lookahead
    # min_periods ensures we don't produce RSI until we have enough history
    roll_up = up.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    roll_down = down.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    rs = roll_up / roll_down

    # Handle division by zero: if roll_down == 0 and roll_up == 0 -> RSI = 50 (no movement)
    # if roll_down == 0 and roll_up > 0 -> RSI = 100
    rsi = pd.Series(index=close.index, dtype=float)
    # Where both are zero
    both_zero = (roll_up == 0) & (roll_down == 0)
    rsi.loc[both_zero] = 50.0

    # Where roll_down == 0 but roll_up > 0
    up_nonzero_down_zero = (roll_down == 0) & (roll_up > 0)
    rsi.loc[up_nonzero_down_zero] = 100.0

    # Where roll_down != 0 compute normally
    valid = roll_down != 0
    rsi.loc[valid] = 100.0 - (100.0 / (1.0 + rs.loc[valid]))

    # rsi will remain NaN for initial warmup bars where min_periods not met

    # Detect crosses (use previous value -> no lookahead)
    rsi_prev = rsi.shift(1)

    enter = (rsi_prev >= oversold) & (rsi < oversold)
    exit = (rsi_prev <= overbought) & (rsi > overbought)

    # Replace NaN booleans with False
    enter = enter.fillna(False).astype(bool)
    exit = exit.fillna(False).astype(bool)

    # Build position series iteratively to ensure no double entries/exits
    idx = close.index
    pos_arr = np.zeros(len(idx), dtype=np.int8)
    state = 0  # 0 = flat, 1 = long

    # Use .iat for fastest access on boolean Series
    enter_vals = enter.values
    exit_vals = exit.values

    for i in range(len(idx)):
        if enter_vals[i] and state == 0:
            state = 1
        elif exit_vals[i] and state == 1:
            state = 0
        pos_arr[i] = state

    position = pd.Series(pos_arr, index=idx, name="position")

    # Ensure dtype is int and no NaNs (fill any remaining NaNs with 0)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
