from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Wilder's RSI for a price series.

    Uses exponential weighted moving average with alpha=1/period (Wilder smoothing).

    Args:
        close: Price series (pd.Series) indexed by timestamp.
        period: RSI lookback period.

    Returns:
        pd.Series of RSI values (0-100) aligned with the input index.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Ensure numeric dtype
    close = pd.to_numeric(close, errors="coerce")

    # Price changes
    delta = close.diff()

    # Gains and losses
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use Wilder's smoothing via ewm with alpha=1/period
    # min_periods=period to avoid producing RSI during warmup
    ma_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Prevent division by zero: where ma_down == 0, set RS to +inf -> RSI=100
    rs = ma_up / ma_down

    rsi = 100 - (100 / (1 + rs))

    # Where both ma_up and ma_down are zero (no price movement), RSI is 50 (neutral)
    both_zero = (ma_up == 0) & (ma_down == 0)
    rsi = rsi.mask(both_zero, 50.0)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute RSI (period=14 by default)
    - Go long when RSI crosses below 30 (oversold)
    - Exit long when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dict containing market data. Must contain key 'ohlcv' with a DataFrame that
              has a 'close' column (pd.Series or single-column DataFrame).
        params: Parameter dict (optional). Supported keys:
            - 'rsi_period' (int): RSI lookback period (default 14).

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (0 or 1) indexed
        the same as the input close series.
    """
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict with key 'ohlcv' containing a DataFrame")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"]

    # If close is a DataFrame (multiple columns), assume single asset and take the first column
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            raise ValueError("ohlcv['close'] DataFrame has no columns")
        # select first column
        close_series = close.iloc[:, 0]
    else:
        close_series = close

    # Ensure we have a time index and numeric series
    close_series = pd.to_numeric(close_series, errors="coerce")

    # Parameters
    rsi_period = int(params.get("rsi_period", 14)) if isinstance(params, dict) else 14
    rsi_lower = 30.0
    rsi_upper = 70.0

    # Compute RSI
    rsi = _compute_rsi(close_series, period=rsi_period)

    # Build crossing masks
    prev_rsi = rsi.shift(1)

    entry_mask = (rsi < rsi_lower) & (prev_rsi >= rsi_lower)
    exit_mask = (rsi > rsi_upper) & (prev_rsi <= rsi_upper)

    # Avoid generating signals during warmup where RSI is NaN
    entry_mask = entry_mask & rsi.notna()
    exit_mask = exit_mask & rsi.notna()

    # Construct position series (0 = flat, 1 = long)
    positions = pd.Series(0, index=close_series.index, dtype=int)

    long = False
    # Iterate through index to maintain state machine (safe for NaNs and repeated signals)
    for i, idx in enumerate(close_series.index):
        # If RSI is NaN at this bar, keep previous position (don't change state)
        if pd.isna(rsi.iloc[i]):
            positions.iloc[i] = 1 if long else 0
            continue

        if (not long) and entry_mask.iloc[i]:
            long = True
        elif long and exit_mask.iloc[i]:
            long = False

        positions.iloc[i] = 1 if long else 0

    # Ensure positions align with original close (in case we selected a single column)
    positions.index = close_series.index

    return {"ohlcv": positions}
