from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having 'close' column. A pd.DataFrame may also be
              passed directly (treated as the 'ohlcv' DataFrame).
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series (values 0 or 1).

    Notes:
        - Only uses 'close' column from the provided DataFrame.
        - Implements a state machine to ensure no double entries/exits.
        - Uses Wilder's smoothing via ewm (alpha=1/period, adjust=False) to
          compute average gains/losses (no lookahead).
    """

    # Allow passing DataFrame directly for convenience
    if not isinstance(data, dict):
        if isinstance(data, pd.DataFrame):
            data = {"ohlcv": data}
        else:
            raise TypeError("data must be a dict mapping 'ohlcv' to a DataFrame or a DataFrame")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame containing 'close' column")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].astype(float).copy()

    # Extract and validate parameters (only use declared params)
    if params is None:
        params = {}

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic validation / clamping to PARAM_SCHEMA bounds
    rsi_period = max(2, min(100, rsi_period))
    oversold = max(0.0, min(50.0, oversold))
    overbought = max(50.0, min(100.0, overbought))

    n = len(close)
    # If empty, return empty series with same index
    if n == 0:
        return {"ohlcv": pd.Series(dtype="int64")}

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period, adjust=False)
    # This uses only past data (no lookahead).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use ewm with alpha=1/period to approximate Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Prepare RSI series handling edge cases
    rs = avg_gain / avg_loss
    rsi = pd.Series(index=close.index, dtype=float)

    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    mask_loss_zero = (avg_loss == 0) & (~mask_both_zero)
    mask_gain_zero = (avg_gain == 0) & (~mask_both_zero)
    mask_regular = ~(mask_both_zero | mask_loss_zero | mask_gain_zero)

    rsi.loc[mask_both_zero] = 50.0
    rsi.loc[mask_loss_zero] = 100.0
    rsi.loc[mask_gain_zero] = 0.0
    # Compute regular RSI values where denom != 0
    rsi.loc[mask_regular] = 100.0 - (100.0 / (1.0 + rs.loc[mask_regular]))

    # There may be NaNs at the beginning if close.diff() produced NaN; keep them
    # but ensure position output does not contain NaN after warmup period by
    # using a well-defined state machine that outputs 0/1 only.

    # Generate signals via a simple state machine to avoid double entries/exits
    position = np.zeros(n, dtype="int8")
    state = 0  # 0 = flat, 1 = long

    # Iterate through time, using only past and current values (no future info)
    # Use iloc-based access to be robust to arbitrary index types
    for i in range(n):
        if i == 0:
            # At the first bar, remain flat unless RSI is already oversold
            # If RSI is below oversold at first valid value, enter immediately
            curr_rsi = rsi.iloc[i]
            if pd.notna(curr_rsi) and curr_rsi < oversold:
                state = 1
            position[i] = state
            continue

        prev_rsi = rsi.iloc[i - 1]
        curr_rsi = rsi.iloc[i]

        # If we don't have valid RSI values, maintain current state
        if pd.isna(prev_rsi) or pd.isna(curr_rsi):
            position[i] = state
            continue

        # Entry: RSI crosses below oversold (or equals then goes below)
        entered = False
        if state == 0:
            # Crossing detection: prev >= oversold and curr < oversold
            if (prev_rsi >= oversold) and (curr_rsi < oversold):
                state = 1
                entered = True
            # Also allow immediate entry if RSI is below oversold and prev RSI was NaN
            # (handled by NaN branch above), or if prev_rsi < oversold but we were flat
            # (this avoids missing entries when the series starts already below threshold)
            elif (prev_rsi < oversold) and (curr_rsi < oversold):
                # If both prev and curr are below oversold and we are flat, enter once
                state = 1
                entered = True

        # Exit: RSI crosses above overbought
        if state == 1 and not entered:
            if (prev_rsi <= overbought) and (curr_rsi > overbought):
                state = 0

        position[i] = state

    pos_series = pd.Series(position.astype("int8"), index=close.index)

    return {"ohlcv": pos_series}
