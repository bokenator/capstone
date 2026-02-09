from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position targets for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute RSI with period `rsi_period` (default 14)
    - Go long when RSI crosses below `oversold` (default 30)
    - Exit long when RSI crosses above `overbought` (default 70)
    - Long-only, single-asset

    Args:
        data: A dictionary containing an 'ohlcv' key mapped to a pandas DataFrame
              that must include a 'close' column.
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'oversold' (float): Oversold threshold. Default 30.
            - 'overbought' (float): Overbought threshold. Default 70.

    Returns:
        A dictionary with key 'ohlcv' containing a pandas Series of position targets
        aligned with the input close prices. Values are 1 for long and 0 for flat.
    """
    # Basic validation
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dict mapping strings to pandas DataFrames")
    if "ohlcv" not in data:
        raise ValueError("`data` must contain an 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("`data['ohlcv']` must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("`data['ohlcv']` must contain a 'close' column")

    close = ohlcv["close"].copy()

    # Parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    oversold = float(params.get("oversold", 30.0)) if params is not None else 30.0
    overbought = float(params.get("overbought", 70.0)) if params is not None else 70.0

    # Compute RSI using vectorbt's indicator
    # vbt.RSI.run returns an object with attribute `.rsi` containing the series
    rsi_out = vbt.RSI.run(close, window=rsi_period)
    rsi = getattr(rsi_out, "rsi", rsi_out)

    # If vectorbt returned a DataFrame (multi-column), reduce to a single series
    if isinstance(rsi, pd.DataFrame):
        if rsi.shape[1] == 1:
            rsi = rsi.iloc[:, 0]
        else:
            # Strategy expects a single asset
            raise ValueError("RSI indicator returned multiple columns; expected single-asset data")

    # Ensure index alignment with close
    rsi = rsi.reindex(close.index)

    # Define crossing conditions
    prev_rsi = rsi.shift(1)

    # Entry: crosses below oversold (previous >= oversold, current < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: crosses above overbought (previous <= overbought, current > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series: we're long when number of entries so far > number of exits so far
    # This handles multiple entry/exit cycles cleanly.
    entries_cs = entries.cumsum()
    exits_cs = exits.cumsum()

    position_bool = entries_cs > exits_cs

    # Convert to integer position (1 = long, 0 = flat), fill any NaNs with 0
    position = position_bool.fillna(False).astype(int)
    position.index = close.index

    return {"ohlcv": position}
