from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price close series.
        period: Lookback period for RSI.

    Returns:
        RSI series (0-100), aligned with close index. Values for the first bar(s) where calculation
        is not possible are set to NaN.
    """
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing (alpha = 1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss is zero, RS is infinite => RSI = 100. Where avg_gain is zero and avg_loss >0 => RSI=0
    rsi = rsi.fillna(0.0)
    rsi[avg_loss == 0] = 100.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset, long-only RSI mean reversion strategy.

    Strategy rules:
      - RSI period = 14
      - Enter long when RSI crosses below 30 (oversold)
      - Exit long when RSI crosses above 70 (overbought)

    Args:
        data: Dict containing at least the 'ohlcv' DataFrame with a 'close' column.
        params: Parameter dict (not used for this fixed-strategy but accepted for compatibility).

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of position targets (1 for long, 0 for flat).

    Notes:
        - Handles NaN values and warmup by not generating entries/exits until RSI is defined.
        - Ensures the returned Series is aligned with the input close price index.
    """
    # Validate input
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing an 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")

    close = ohlcv["close"].copy()

    # Parameters (fixed for this prompt, but allow overrides via params)
    rsi_period = int(params.get("rsi_period", 14)) if isinstance(params, dict) else 14
    rsi_entry_level = float(params.get("entry_level", 30.0)) if isinstance(params, dict) else 30.0
    rsi_exit_level = float(params.get("exit_level", 70.0)) if isinstance(params, dict) else 70.0

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crossings
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below entry_level: previous >= entry_level and current < entry_level
    entry_signals = (prev_rsi >= rsi_entry_level) & (rsi < rsi_entry_level)

    # Exit when RSI crosses above exit_level: previous <= exit_level and current > exit_level
    exit_signals = (prev_rsi <= rsi_exit_level) & (rsi > rsi_exit_level)

    # Replace NaN booleans with False
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series: 0 flat, 1 long. Single-asset, no pyramiding.
    position = pd.Series(index=close.index, dtype=float)
    position.iloc[:] = 0.0

    in_position = False
    for idx in close.index:
        if not in_position:
            # If entry signal on this bar, enter at next bar (but in this simple runner entries are taken on the same bar)
            if entry_signals.loc[idx]:
                in_position = True
                position.loc[idx] = 1.0
            else:
                position.loc[idx] = 0.0
        else:
            # Currently in position, check exit
            if exit_signals.loc[idx]:
                in_position = False
                position.loc[idx] = 0.0
            else:
                position.loc[idx] = 1.0

    # Ensure integer-ish values 0/1 and align dtypes
    position = position.fillna(0.0).astype(float)

    return {"ohlcv": position}
