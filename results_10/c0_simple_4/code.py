import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's RSI using exponential moving average (Wilder smoothing).

    Args:
        close: Price series
        period: RSI lookback period

    Returns:
        RSI series aligned with close.index
    """
    close = close.astype(float).copy()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing via EWM with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "rsi"
    return rsi


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate position series for an RSI mean-reversion strategy.

    Strategy rules:
    - RSI(14)
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only: positions are 0 (flat) or 1 (long)

    Args:
        data: Dictionary containing 'ohlcv' DataFrame with a 'close' column
        params: Optional parameters (supported keys: 'rsi_period', 'rsi_lower', 'rsi_upper')

    Returns:
        A dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1) indexed like the input close
    """
    # Validate input
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Ensure datetime index where possible
    if not pd.api.types.is_datetime64_any_dtype(close.index):
        try:
            close.index = pd.to_datetime(close.index)
        except Exception:
            # If conversion fails, continue with original index
            pass

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    rsi_lower = float(params.get("rsi_lower", 30.0)) if params is not None else 30.0
    rsi_upper = float(params.get("rsi_upper", 70.0)) if params is not None else 70.0

    # Compute RSI (try vectorbt first, fallback to manual implementation)
    try:
        rsi = vbt.RSI.run(close, window=rsi_period).rsi
        # If vbt returns a DataFrame for some reason, squeeze to Series
        if isinstance(rsi, pd.DataFrame):
            if rsi.shape[1] == 1:
                rsi = rsi.iloc[:, 0]
            else:
                raise ValueError("vbt.RSI returned a multi-column DataFrame; expected single series input")
    except Exception:
        rsi = _compute_rsi(close, period=rsi_period)

    # Align RSI with close index
    rsi = rsi.reindex(close.index)

    # Avoid signals during warmup
    valid = rsi.notna()
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below lower threshold
    entries = ((prev_rsi >= rsi_lower) & (rsi < rsi_lower)) & valid
    # Exit: RSI crosses above upper threshold
    exits = ((prev_rsi <= rsi_upper) & (rsi > rsi_upper)) & valid

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Build positions from entry/exit signals
    signal = entries.astype(int) - exits.astype(int)
    cumsum = signal.cumsum().clip(lower=0)
    positions = (cumsum > 0).astype(int)
    positions = pd.Series(positions, index=close.index, name="position")

    # Do not hold positions when price or RSI is NaN
    positions[close.isna()] = 0
    positions[rsi.isna()] = 0

    # Ensure first bar is flat so the backtester can detect the initial entry
    if len(positions) > 0:
        positions.iloc[0] = 0

    return {"ohlcv": positions}
