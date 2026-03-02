from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean reversion strategy.

    Strategy logic
    - RSI period = 14
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dictionary containing OHLCV data under the key 'ohlcv'. The value
              should be a pandas.DataFrame with a 'close' column.
        params: Strategy parameters (not used for this fixed-rule strategy,
                included for compatibility).

    Returns:
        A dict with key 'ohlcv' mapping to a pandas.Series of position targets
        where 1 means long and 0 means flat. The series index matches the
        input close price index.
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("'ohlcv' DataFrame must contain a 'close' column")

    close = ohlcv["close"].copy()
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=ohlcv.index)

    # Compute RSI using vectorbt's indicator (window 14)
    # vbt.RSI.run returns an indicator object with output 'rsi'
    rsi_ind = vbt.RSI.run(close, window=14)
    rsi = rsi_ind.rsi

    # Helper to squeeze outputs to a 1D Series for single-asset case
    def _to_1d_series(x: Any, reference_index: pd.Index) -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.copy()
        elif isinstance(x, pd.DataFrame):
            # If DataFrame has a single column, take it. Otherwise, try to squeeze.
            if x.shape[1] == 1:
                s = x.iloc[:, 0].copy()
            else:
                # Fallback: take the first column
                s = x.iloc[:, 0].copy()
        elif isinstance(x, np.ndarray):
            s = pd.Series(x, index=reference_index)
        else:
            # Try to convert
            s = pd.Series(x, index=reference_index)

        # Ensure index alignment
        if not s.index.equals(reference_index):
            s.index = reference_index
        return s

    rsi = _to_1d_series(rsi, close.index)

    # Use vectorbt indicator crossing helpers where available
    # These return booleans where the crossing event happens
    try:
        entries = rsi_ind.rsi_crossed_below(30)
        exits = rsi_ind.rsi_crossed_above(70)
    except Exception:
        # Fallback to manual crossing detection if helper not available
        entries = (rsi < 30) & (rsi.shift(1) >= 30)
        exits = (rsi > 70) & (rsi.shift(1) <= 70)

    entries = _to_1d_series(entries, close.index).fillna(False).astype(bool)
    exits = _to_1d_series(exits, close.index).fillna(False).astype(bool)

    # Build position series: 1 when in a long trade, 0 otherwise.
    # We use cumulative counts of entries and exits to determine when we're long.
    entry_cum = entries.astype(int).cumsum()
    exit_cum = exits.astype(int).cumsum()

    pos_bool = (entry_cum - exit_cum) > 0
    position = pos_bool.astype(int)
    position.index = close.index

    # Ensure no NaNs and dtype is integer (0 or 1)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
