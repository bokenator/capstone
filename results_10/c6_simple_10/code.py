# Complete implementation for RSI mean reversion signal generator
from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period, adjust=False).

    This implementation is causal (no lookahead) and returns a pd.Series aligned
    with the input index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (EWMA with alpha = 1/period)
    # Set min_periods=period so initial values are NaN until enough data points
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero: if avg_loss == 0 -> RSI = 100
    rsi = rsi.where(avg_loss != 0, 100.0)

    # If both avg_gain and avg_loss are zero, define RSI as 50 (neutral)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    rsi.name = "rsi"
    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Args:
        data: dict with key "ohlcv" containing a DataFrame with a 'close' column.
        params: dict containing 'rsi_period', 'oversold', 'overbought'.

    Returns:
        dict with key "ohlcv" mapping to a pd.Series of positions (0 or 1).
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].copy()

    # Read parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if len(close) == 0:
        # Return empty series with same index
        return {"ohlcv": pd.Series(dtype=float, index=close.index)}

    # Compute RSI (causal)
    rsi = _compute_rsi(close, period=rsi_period)

    # Define crossing conditions (use previous bar to detect cross)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold
    potential_entries = (rsi < oversold) & (prev_rsi >= oversold)

    # Exit: RSI crosses above overbought
    potential_exits = (rsi > overbought) & (prev_rsi <= overbought)

    # Replace NaNs in signals with False to avoid accidental triggers
    potential_entries = potential_entries.fillna(False)
    potential_exits = potential_exits.fillna(False)

    # Build position series sequentially to avoid double entries
    pos_values = np.zeros(len(close), dtype=int)
    in_pos = False

    # Use integer indices for performance and determinism
    for i in range(len(close)):
        if in_pos:
            # If currently long, check for exit first
            if potential_exits.iat[i]:
                in_pos = False
                pos_values[i] = 0
            else:
                pos_values[i] = 1
        else:
            # If flat, check for entry
            if potential_entries.iat[i]:
                in_pos = True
                pos_values[i] = 1
            else:
                pos_values[i] = 0

    positions = pd.Series(pos_values, index=close.index, name="position")

    # Ensure positions are integers 0 or 1 and contain no NaN
    positions = positions.astype(int).fillna(0)

    return {"ohlcv": positions}
