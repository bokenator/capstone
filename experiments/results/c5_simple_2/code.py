import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def generate_signals(
    data: dict[str, pd.DataFrame] | pd.DataFrame | pd.Series,
    params: dict[str, Any] | None = None,
) -> dict[str, pd.Series] | Tuple[pd.Series, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Either a dict with key 'ohlcv' mapping to a DataFrame containing 'close',
              or directly a DataFrame/Series of close prices.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        If input is a dict: Dict mapping 'ohlcv' to position Series (+1 long, 0 flat, -1 short)
        If input is DataFrame/Series: returns tuple (entries, exits) where entries and exits
        are boolean Series suitable for backtesting in simple tests.

    Notes:
        - Calculation uses Wilder's RSI (EMA with alpha=1/period, adjust=False)
        - No lookahead: signals at time t depend only on data up to t.
        - Handles NaNs by treating them as no-signal; outputs contain no NaNs (positions filled with 0).
    """

    # Validate and normalize params
    if params is None:
        params = {}
    rsi_period = int(params.get("rsi_period", 14))
    # enforce bounds (as in PARAM_SCHEMA)
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

    # Extract close price series depending on input type
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("Input dict must contain 'ohlcv' key with OHLCV DataFrame")
        df = data["ohlcv"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("data['ohlcv'] must be a pandas DataFrame")
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        close = pd.to_numeric(df["close"], errors="coerce")
    elif isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        close = pd.to_numeric(data["close"], errors="coerce")
    elif isinstance(data, pd.Series):
        close = pd.to_numeric(data, errors="coerce")
    else:
        raise ValueError("Unsupported data type. Provide dict, DataFrame, or Series")

    # Compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    delta = close.diff()
    # Replace initial NaN delta with 0 to avoid propagation
    delta = delta.fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(0.0)  # if avg_loss and avg_gain are zero -> set RSI to 0

    # Detect cross below oversold (entry) and cross above overbought (exit)
    prev_rsi = rsi.shift(1)

    entry_signal = (rsi < oversold) & (prev_rsi >= oversold)
    exit_signal = (rsi > overbought) & (prev_rsi <= overbought)

    # Replace NaNs in signals with False (no signal)
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series iteratively to prevent double-entries and ensure no lookahead
    pos = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    for idx in close.index:
        if not in_position:
            if entry_signal.loc[idx]:
                pos.loc[idx] = 1
                in_position = True
            else:
                pos.loc[idx] = 0
        else:
            # Currently long
            if exit_signal.loc[idx]:
                pos.loc[idx] = 0
                in_position = False
            else:
                pos.loc[idx] = 1

    # Ensure no NaN values anywhere
    pos = pos.fillna(0).astype(int)

    # For convenience in some unit tests, if input was Series/DataFrame, return entries/exits
    if not isinstance(data, dict):
        # entries/exits as boolean Series where an entry is a transition from 0 to 1
        position_diff = pos.diff().fillna(0)
        entries = position_diff > 0
        exits = position_diff < 0
        return entries, exits

    # If input was dict, return dict with 'ohlcv' key
    return {"ohlcv": pos}
