# Complete implementation of RSI mean reversion signal generator
from __future__ import annotations
from typing import Any, Optional, Tuple, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series aligned with close index.
    """
    # Ensure float
    close = close.astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing via ewm with alpha=1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # RSI is NaN where avg_loss is zero and avg_gain is zero initially
    return rsi


def generate_signals(
    input_data: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Generate RSI mean-reversion signals.

    Behavior depends on input type:
    - If input_data is a dict with key 'ohlcv' (DataFrame), function returns a dict
      with key 'ohlcv' containing a position series (1 for long, 0 for flat).
      Signature: generate_signals(data_dict, params) -> { 'ohlcv': pd.Series }

    - If input_data is a pd.Series or pd.DataFrame representing prices, function
      returns a tuple (entries, exits) where entries/exits are boolean pd.Series.
      Signature: entries, exits = generate_signals(prices_series)

    Strategy logic:
    - RSI period = 14
    - Enter long when RSI crosses below 30
    - Exit when RSI crosses above 70
    - Long-only, do not enter when already long

    Args:
        input_data: price series, price DataFrame with 'close', or data dict used by
            the backtest runner.
        params: optional params dict (ignored currently, kept for compatibility)

    Returns:
        dict or tuple depending on input type (see above).
    """
    period = 14

    # Extract close series depending on input type
    close: pd.Series
    returned_as_dict = False

    if isinstance(input_data, dict):
        # Expect data['ohlcv'] DataFrame with 'close' column
        if "ohlcv" not in input_data:
            raise ValueError("input_data dict must contain 'ohlcv' key with OHLCV DataFrame")
        ohlcv = input_data["ohlcv"]
        if isinstance(ohlcv, pd.DataFrame) and "close" in ohlcv.columns:
            close = ohlcv["close"].copy()
        else:
            raise ValueError("input_data['ohlcv'] must be a DataFrame with a 'close' column")
        returned_as_dict = True
    elif isinstance(input_data, pd.DataFrame):
        # If DataFrame has 'close' use it, otherwise try first column
        if "close" in input_data.columns:
            close = input_data["close"].copy()
        else:
            close = input_data.iloc[:, 0].copy()
    elif isinstance(input_data, pd.Series):
        close = input_data.copy()
    else:
        # Try to convert to Series
        try:
            close = pd.Series(input_data)
        except Exception:
            raise ValueError("Unsupported input_data type for generate_signals")

    # Ensure index monotonic and preserved
    close.index = pd.Index(close.index)

    # Compute RSI
    rsi = _compute_rsi(close, period=period)

    # Define crossing conditions using previous value to avoid lookahead
    rsi_prev = rsi.shift(1)

    # Entry: previous >=30 and current <30 (cross below 30)
    raw_entry = (rsi_prev >= 30) & (rsi < 30)

    # Exit: previous <=70 and current >70 (cross above 70)
    raw_exit = (rsi_prev <= 70) & (rsi > 70)

    # Replace NaN with False for signal arrays
    raw_entry = raw_entry.fillna(False)
    raw_exit = raw_exit.fillna(False)

    # Build position series by iterating to enforce no double entries and long-only
    position = pd.Series(index=close.index, dtype=float)
    in_pos = False

    for idx in close.index:
        if raw_entry.loc[idx] and not in_pos:
            in_pos = True
            position.loc[idx] = 1.0
            continue
        if raw_exit.loc[idx] and in_pos:
            in_pos = False
            position.loc[idx] = 0.0
            continue
        # Otherwise carry forward previous position or set 0 if first
        if position.index.get_loc(idx) == 0:
            position.loc[idx] = 1.0 if in_pos else 0.0
        else:
            # Use previous value
            prev_idx = position.index[position.index.get_loc(idx) - 1]
            position.loc[idx] = position.loc[prev_idx]

    # Ensure dtype int-like 0/1 and no NaNs
    position = position.fillna(0.0).astype(int)

    # Compute entries/exits from position diffs (boolean Series)
    pos_diff = position.diff().fillna(0).astype(int)
    entries = pos_diff > 0
    exits = pos_diff < 0

    # Ensure boolean dtype and same index
    entries = entries.astype(bool)
    exits = exits.astype(bool)

    # After warmup period (50), there should be no NaNs in returned series.
    # We ensured no NaNs above. For safety, clip index and fillna False on entries/exits.
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    if returned_as_dict:
        # Return the position series in expected structure for the backtest runner
        return {"ohlcv": position}

    # If single-series input, return entries and exits tuple (as tests expect)
    return entries, exits


# If this module is run directly, provide a tiny smoke test
if __name__ == "__main__":
    # Basic sanity check
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(np.linspace(100, 150, 100), index=dates)
    e, x = generate_signals(prices)
    print("Entries:", e.sum(), "Exits:", x.sum())
