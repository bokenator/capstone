"""
RSI Mean Reversion Signal Generator

Implements a simple RSI mean-reversion strategy:
- RSI period configurable (default 14)
- Go long when RSI crosses below oversold threshold (e.g., 30)
- Exit when RSI crosses above overbought threshold (e.g., 70)

Exports:
- generate_signals(data: dict, params: dict) -> dict

The returned dict has key "ohlcv" mapping to a pd.Series of 0/1 position
values aligned with data["ohlcv"].index.

Uses vectorbt's RSI indicator for correctness. Verified vbt.RSI.run API via
vectorbt docs.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

import vectorbt as vbt


def generate_signals(data: dict, params: dict) -> dict:
    """
    Generate long-only position targets (0 or 1) based on RSI mean reversion.

    Args:
        data (dict): Must contain "ohlcv" DataFrame with a 'close' column.
        params (dict): Must contain keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        dict: {"ohlcv": pd.Series} where the Series contains 0 (flat) or 1 (long)
              aligned to the input close index.
    """
    # Validate input
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict with key 'ohlcv' containing a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame with a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with basic validation
    try:
        rsi_period = int(params.get("rsi_period", 14))
    except Exception:
        rsi_period = 14
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    try:
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception:
        oversold = 30.0
        overbought = 70.0

    if oversold >= overbought:
        raise ValueError("oversold threshold must be less than overbought threshold")

    # Compute RSI using vectorbt's implementation (no lookahead)
    # vbt.RSI.run returns an object with attribute 'rsi' (pd.Series)
    rsi_ind = vbt.RSI.run(close, window=rsi_period)
    rsi = rsi_ind.rsi

    # Ensure rsi is a Series aligned with close
    if isinstance(rsi, pd.DataFrame):
        # If multi-column, take the first column
        rsi = rsi.iloc[:, 0]
    rsi = rsi.reindex(close.index)

    # Prepare numpy arrays for fast iteration without lookahead
    idx = close.index
    n = len(idx)
    rsi_vals = rsi.to_numpy(dtype=float)

    pos_arr = np.zeros(n, dtype=int)
    prev_pos = 0

    # Iterate sequentially; decisions at time t depend only on rsi up to t
    for i in range(n):
        entry = False
        exit = False

        # Need previous rsi to detect crossing
        if i >= 1:
            prev_r = rsi_vals[i - 1]
            curr_r = rsi_vals[i]

            # Only evaluate crossing if both values are finite
            if np.isfinite(prev_r) and np.isfinite(curr_r):
                if (prev_r >= oversold) and (curr_r < oversold):
                    entry = True
                if (prev_r <= overbought) and (curr_r > overbought):
                    exit = True

        # Apply state machine: only enter if flat, only exit if long
        if entry and prev_pos == 0:
            prev_pos = 1
        elif exit and prev_pos == 1:
            prev_pos = 0

        pos_arr[i] = prev_pos

    position = pd.Series(pos_arr, index=idx)

    return {"ohlcv": position}
