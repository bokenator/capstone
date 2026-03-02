import pandas as pd
import numpy as np
from typing import Dict


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
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float)

    # Validate and extract params
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Helper: compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute RSI for a price series.

        Uses Wilder's smoothing (exponential moving average with alpha=1/period).
        Returns a Series with the same index as input. Early values will be NaN
        until there is enough data for the Wilder average.
        """
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)

        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)

        # Wilder's smoothing
        ma_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        ma_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Avoid division by zero; result will be inf where ma_down == 0 and ma_up > 0
        rs = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Replace infinite values (when ma_down == 0) with 100
        rsi = rsi.replace([np.inf, -np.inf], np.nan)

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Build entry/exit signals based on RSI crosses
    prev_rsi = rsi.shift(1)

    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs (from warmup) with False (no signal)
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Convert boolean signals to a position series (0 or 1) - long-only
    n = len(close)
    pos_vals = np.zeros(n, dtype=int)
    in_position = False

    # Use iloc-based iteration for performance and to preserve index alignment
    for i in range(n):
        if not in_position and bool(entry_signals.iloc[i]):
            # Enter long
            in_position = True
            pos_vals[i] = 1
        elif in_position:
            if bool(exit_signals.iloc[i]):
                # Exit long
                in_position = False
                pos_vals[i] = 0
            else:
                # Stay long
                pos_vals[i] = 1
        else:
            # Stay flat
            pos_vals[i] = 0

    position = pd.Series(pos_vals, index=close.index, name="position")

    return {"ohlcv": position}
