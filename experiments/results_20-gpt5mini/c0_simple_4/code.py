from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price close series.
        period: RSI lookback period (must be >= 1).

    Returns:
        pd.Series: RSI values aligned to the input index. Initial values are NaN
        until enough data (min_periods=period).
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: exponential with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # RSI formula
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Generate long-only signals based on RSI mean reversion.

    Strategy rules:
    - Compute RSI with period (default 14).
    - Go long when RSI crosses below the oversold threshold (default 30).
    - Exit long when RSI crosses above the overbought threshold (default 70).

    Args:
        data: Dict containing market data. Must include key "ohlcv" mapping to a
            pandas DataFrame (or Series) with a "close" column/series.
        params: Dict of strategy parameters. Supported keys:
            - "rsi_period" (int, default 14)
            - "rsi_oversold" (float, default 30)
            - "rsi_overbought" (float, default 70)

    Returns:
        Dict with at least the key "ohlcv" containing a pandas Series of position
        targets: 1 for long, 0 for flat. The Series index matches input close.

    Raises:
        ValueError / KeyError / TypeError for invalid inputs.
    """
    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with an 'ohlcv' entry")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV data")

    ohlcv = data["ohlcv"]

    # Extract close series
    if isinstance(ohlcv, pd.DataFrame):
        if "close" not in ohlcv.columns:
            raise KeyError("'ohlcv' DataFrame must contain a 'close' column")
        close = ohlcv["close"].copy()
    elif isinstance(ohlcv, pd.Series):
        close = ohlcv.copy()
    else:
        raise TypeError("'ohlcv' must be a pandas DataFrame or Series")

    if not isinstance(close.index, pd.Index):
        # Ensure index is a pandas Index for alignment
        close.index = pd.Index(close.index)

    if close.dropna().empty:
        raise ValueError("close series is empty or only NaNs")

    # Parameters with sensible defaults
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    oversold = float(params.get("rsi_oversold", 30)) if params is not None else 30.0
    overbought = float(params.get("rsi_overbought", 70)) if params is not None else 70.0

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    if not (0 <= oversold < overbought <= 100):
        # Correct parameters if given incorrectly
        raise ValueError("Expect 0 <= rsi_oversold < rsi_overbought <= 100")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (from >= oversold to < oversold)
    entries = (prev_rsi >= oversold) & (rsi < oversold) & rsi.notna() & prev_rsi.notna()

    # Exit: RSI crosses above overbought (from <= overbought to > overbought)
    exits = (prev_rsi <= overbought) & (rsi > overbought) & rsi.notna() & prev_rsi.notna()

    # Build position series: 1 = long, 0 = flat
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate in chronological order
    for ts in close.index:
        if in_position:
            # Check exit first while in position
            if exits.loc[ts]:
                in_position = False
        else:
            # Check entry when flat
            if entries.loc[ts]:
                in_position = True

        position.loc[ts] = 1 if in_position else 0

    # Ensure the output Series has the same name and index dtype as close
    position.name = "position"

    return {"ohlcv": position, "rsi": rsi}
