import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column. Alternatively, a single
              pd.DataFrame or pd.Series may be provided and will be interpreted
              as the 'ohlcv' close series.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series. Position values: +1 (long), 0 (flat), -1 (short)
    """
    # Validate and normalize inputs to obtain a close price Series
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data dict must contain 'ohlcv' key with a DataFrame")
        ohlcv = data["ohlcv"]
        if not isinstance(ohlcv, pd.DataFrame):
            raise ValueError("data['ohlcv'] must be a pandas DataFrame")
        if "close" not in ohlcv.columns:
            raise ValueError("ohlcv DataFrame must contain 'close' column")
        close = ohlcv["close"]
    elif isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        close = data["close"]
    elif isinstance(data, pd.Series):
        close = data
    else:
        raise ValueError("data must be a dict, DataFrame, or Series containing close prices")

    # Extract parameters with defaults and validation
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    # Compute RSI using vectorbt (module-qualified call)
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Previous RSI (t-1) using pandas API in module-qualified form
    prev_rsi = pd.Series.shift(rsi, 1)

    # Define crossing conditions (cross below oversold -> entry; cross above overbought -> exit)
    # Fill NA in prev_rsi with extremes so that initial bars do not trigger crossings
    prev_rsi_filled_for_entry = pd.Series.fillna(prev_rsi, np.inf)
    prev_rsi_filled_for_exit = pd.Series.fillna(prev_rsi, -np.inf)

    entry_mask = (prev_rsi_filled_for_entry >= oversold) & (rsi < oversold)
    exit_mask = (prev_rsi_filled_for_exit <= overbought) & (rsi > overbought)

    # Ensure boolean series and align index
    entry_mask = pd.Series(entry_mask.values, index=close.index)
    exit_mask = pd.Series(exit_mask.values, index=close.index)

    # Build position series by walking through time to maintain state (no double-entry)
    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)
    in_position = False

    for i in range(n):
        if not in_position:
            # Enter when entry condition met
            if bool(entry_mask.iloc[i]):
                in_position = True
                pos_arr[i] = 1
            else:
                pos_arr[i] = 0
        else:
            # Currently long: stay long unless exit condition met
            if bool(exit_mask.iloc[i]):
                in_position = False
                pos_arr[i] = 0
            else:
                pos_arr[i] = 1

    position_series = pd.Series(pos_arr, index=close.index)

    return {"ohlcv": position_series}
