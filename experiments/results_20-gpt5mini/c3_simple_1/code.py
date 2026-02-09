import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    This implementation is causal (no lookahead): it uses exponential
    smoothing with adjust=False which depends only on past data.

    Args:
        close: Series of close prices.
        period: RSI period (e.g., 14).

    Returns:
        RSI series aligned with `close` index.
    """
    if period <= 0:
        raise ValueError("rsi period must be positive")

    close = close.astype(float).copy()

    # If all values are NaN, return a Series of NaN
    if close.isna().all():
        return pd.Series(np.nan, index=close.index)

    # Fill price NaNs conservatively (forward then backward) so indicators compute
    close = close.ffill().bfill()

    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder's smoothing (exponential with alpha=1/period, causal)
    alpha = 1.0 / float(period)
    # min_periods=period ensures the initial averages stabilize similarly to classic RSI
    avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases explicitly
    # If both gain and loss are zero -> RSI = 50 (no movement)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)

    # If loss == 0 and gain > 0 -> RSI = 100
    loss_zero_gain_positive = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~loss_zero_gain_positive, 100.0)

    # If gain == 0 and loss > 0 -> RSI = 0
    gain_zero_loss_positive = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.where(~gain_zero_loss_positive, 0.0)

    # For early periods where min_periods not satisfied, fill with neutral 50 to avoid NaNs
    rsi = rsi.fillna(50.0)

    return rsi


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy rules:
    - Compute RSI with period params['rsi_period']
    - Go long when RSI crosses below params['oversold'] (e.g., 30)
    - Exit (go flat) when RSI crosses above params['overbought'] (e.g., 70)

    Args:
        data: dict with key "ohlcv" that is a DataFrame containing a 'close' column.
        params: dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series of 0/1
              values aligned with input close index.
    """
    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict with key 'ohlcv'")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].copy()

    # Ensure index alignment and types
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Extract parameters with safe defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # If there are no bars, return empty series
    if len(close) == 0:
        return {"ohlcv": pd.Series(dtype=int)}

    # Compute RSI (causal)
    rsi = compute_rsi(close, rsi_period)

    # Detect crosses using previous value (no lookahead). Require previous to be valid to avoid
    # spurious signals at the start.
    prev_rsi = rsi.shift(1)

    cross_below = prev_rsi.notna() & (prev_rsi >= oversold) & (rsi < oversold)
    cross_above = prev_rsi.notna() & (prev_rsi <= overbought) & (rsi > overbought)

    # Initialize position array (0 = flat, 1 = long)
    positions = np.zeros(len(close), dtype=int)

    state = 0  # 0 = flat, 1 = long
    # Iterate sequentially to ensure no double-entry and to respect state
    for i in range(len(close)):
        if state == 0:
            if bool(cross_below.iloc[i]):
                state = 1
                positions[i] = 1
            else:
                positions[i] = 0
        else:  # state == 1
            if bool(cross_above.iloc[i]):
                state = 0
                positions[i] = 0
            else:
                positions[i] = 1

    position_series = pd.Series(positions, index=close.index, name="position").astype(int)

    return {"ohlcv": position_series}
