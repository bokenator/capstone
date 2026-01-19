import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Any, Dict, Tuple, Union, Optional


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using vectorbt's RSI indicator.

    Args:
        close: Close price series.
        window: RSI lookback period.

    Returns:
        RSI series aligned with close.index.
    """
    # Use vectorbt RSI implementation (allowed API)
    rsi = vbt.RSI.run(close, window=window).rsi
    return rsi


def generate_signals(
    data: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, pd.Series], Tuple[pd.Series, pd.Series]]:
    """Generate trading signals for an RSI mean-reversion strategy.

    Behavior:
    - If `data` is a dict (expected by the backtest runner), returns a dict with key
      'ohlcv' mapping to a position series (+1 long, 0 flat).
    - If `data` is a pd.Series of close prices (used in unit tests), returns a tuple
      (entries, exits) of boolean Series.

    Strategy logic:
    - RSI period = 14
    - Entry (go long) when RSI crosses below 30 (prev >= 30 and curr < 30)
    - Exit when RSI crosses above 70 (prev <= 70 and curr > 70)
    - Long-only

    Args:
        data: Either a dict with key 'ohlcv' containing a DataFrame with 'close' column,
              or a pd.Series of close prices.
        params: Optional params dict (ignored; kept for API compatibility).

    Returns:
        Dict with 'ohlcv' -> position Series, or (entries, exits) boolean Series.
    """
    # Determine input type
    # Case 1: dict input used by run_backtest
    if isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("Input dict must contain 'ohlcv' key with OHLCV DataFrame")
        ohlcv = data["ohlcv"]
        if "close" not in ohlcv:
            raise ValueError("ohlcv DataFrame must contain 'close' column")
        close = ohlcv["close"]

        # Compute RSI
        rsi = _compute_rsi(close, window=14)

        # Previous RSI (shifted by 1). For initial NaN, fill with current value to avoid
        # spurious cross signals at the very first bar.
        rsi_prev = rsi.shift(1)
        rsi_prev = rsi_prev.fillna(rsi)

        # Entry: prev >= 30 and curr < 30 (cross below 30)
        entries = (rsi_prev >= 30) & (rsi < 30)

        # Exit: prev <= 70 and curr > 70 (cross above 70)
        exits = (rsi_prev <= 70) & (rsi > 70)

        # Build position series (long-only, +1 for long, 0 for flat)
        pos = pd.Series(np.zeros(len(close), dtype=np.int8), index=close.index)
        in_position = False
        # Iterate and set position at each bar (no lookahead because we only use past/current RSI)
        for i in range(len(close)):
            if entries.iloc[i] and not in_position:
                # Enter on this bar
                pos.iloc[i] = 1
                in_position = True
            elif exits.iloc[i] and in_position:
                # Exit on this bar -> position becomes flat (0)
                pos.iloc[i] = 0
                in_position = False
            else:
                # Carry forward previous position (or remain flat at first bar)
                if i > 0:
                    pos.iloc[i] = pos.iloc[i - 1]
                else:
                    pos.iloc[i] = 0

        # Ensure no NaN values (pos is int, but keep defensive filling)
        pos = pos.fillna(0).astype(np.int8)

        return {"ohlcv": pos}

    # Case 2: Series input (unit tests expecting entries, exits)
    if isinstance(data, pd.Series):
        close = data
        rsi = _compute_rsi(close, window=14)
        rsi_prev = rsi.shift(1)
        rsi_prev = rsi_prev.fillna(rsi)

        entries = (rsi_prev >= 30) & (rsi < 30)
        exits = (rsi_prev <= 70) & (rsi > 70)

        # Ensure boolean dtype and aligned index
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries.astype(bool), exits.astype(bool)

    raise ValueError("Unsupported input type for generate_signals")
