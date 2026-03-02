import pandas as pd
import numpy as np
from typing import Any


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    if close.isna().all():
        # Nothing to do if all close prices are NaN
        positions = pd.Series(0, index=close.index, dtype="int64")
        return {"ohlcv": positions}

    # Extract and validate params (only allowed keys)
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid params: {e}")

    # Parameter validation according to PARAM_SCHEMA
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0 and 50")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Helper: compute RSI using Wilder's smoothing (EMA with alpha=1/period)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing: exponential moving average with alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Fill any NaNs (e.g., at the beginning) with neutral 50
        rsi = rsi.fillna(50.0)
        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below the oversold level (from >= oversold to < oversold)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above the overbought level (from <= overbought to > overbought)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series: 1 for long, 0 for flat. Iterate to ensure correct stateful behavior.
    pos_array = np.zeros(len(close), dtype="int64")
    in_long = False

    # Use iloc for positional access to align with entry/exit boolean series
    for i in range(len(close)):
        try:
            is_entry = bool(entry_signals.iloc[i])
        except Exception:
            is_entry = False
        try:
            is_exit = bool(exit_signals.iloc[i])
        except Exception:
            is_exit = False

        if is_entry and not in_long:
            in_long = True
        elif is_exit and in_long:
            in_long = False

        pos_array[i] = 1 if in_long else 0

    positions = pd.Series(pos_array, index=close.index, dtype="int64")

    return {"ohlcv": positions}
