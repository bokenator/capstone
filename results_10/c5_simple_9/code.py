from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(
    data: Dict[str, pd.DataFrame] | pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, pd.Series] | pd.Series:
    """
    Generate position signals for an RSI mean reversion strategy.

    Accepts either a dict with key 'ohlcv' mapping to a DataFrame (preferred),
    or a DataFrame directly (legacy/testing convenience). When a dict is passed,
    returns a dict {"ohlcv": position_series}. When a DataFrame is passed,
    returns the position Series directly.

    Strategy:
    - Compute RSI (Wilder's smoothing) with period = params['rsi_period']
    - Entry: RSI crosses below params['oversold']
    - Exit:  RSI crosses above params['overbought']
    - Long-only: positions are 1 (long) or 0 (flat)

    Args:
        data: Either {'ohlcv': DataFrame} or DataFrame with a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict mapping slot to pd.Series (or pd.Series if DataFrame input).
    """

    # Allow callers to pass the DataFrame directly for convenience/testing
    legacy_return_series = False
    if isinstance(data, pd.DataFrame):
        ohlcv = data
        legacy_return_series = True
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("data dict must contain 'ohlcv' key with a DataFrame value")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("data must be a pandas DataFrame or a dict containing 'ohlcv' DataFrame")

    # Validate close column
    if "close" not in ohlcv.columns:
        raise ValueError("Input 'ohlcv' DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Extract and validate parameters (use only allowed param keys)
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Basic sanity clamps (keep within allowed ranges but do not silently correct wildly invalid values)
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if rsi_period > 100:
        raise ValueError("rsi_period must be <= 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    # Compute RSI using Wilder's smoothing (causal: no lookahead)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm with alpha=1/period and adjust=False is causal
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_values = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases explicitly
    # If both avg_gain and avg_loss are zero -> RSI = 50 (no moves)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi_values = np.where(both_zero, 50.0, rsi_values)
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100 (all gains)
    rsi_values = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi_values)
    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0 (all losses)
    rsi_values = np.where((avg_gain == 0) & (avg_loss > 0), 0.0, rsi_values)

    rsi = pd.Series(rsi_values, index=close.index)

    # Define crossing conditions (use previous RSI; treat missing previous as threshold value to allow
    # immediate entry/exit when the first valid RSI crosses)
    prev_rsi = rsi.shift(1)

    prev_for_entry = prev_rsi.fillna(oversold)
    prev_for_exit = prev_rsi.fillna(overbought)

    entry_signals = (rsi < oversold) & (prev_for_entry >= oversold)
    exit_signals = (rsi > overbought) & (prev_for_exit <= overbought)

    # Ensure signals are boolean Series aligned with index
    entry_signals = entry_signals.astype(bool)
    exit_signals = exit_signals.astype(bool)

    # Build position series: long-only (0 or 1). Iterate once to avoid double-entries.
    n = len(close)
    positions = np.zeros(n, dtype=int)
    state = 0  # 0 = flat, 1 = long

    # Use .to_numpy() for faster iteration
    entry_np = entry_signals.to_numpy(dtype=bool)
    exit_np = exit_signals.to_numpy(dtype=bool)

    for i in range(n):
        if state == 0 and entry_np[i]:
            state = 1
        elif state == 1 and exit_np[i]:
            state = 0
        positions[i] = state

    position_series = pd.Series(positions, index=close.index)

    # Return in requested shape
    if legacy_return_series:
        return position_series

    return {"ohlcv": position_series}
