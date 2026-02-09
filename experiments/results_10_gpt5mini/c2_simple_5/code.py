# RSI mean reversion signal generator
from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI as pd.Series with same index as close. Warmup values are NaN.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (exponential moving average with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Replace infinite values if avg_loss == 0
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position series based on RSI mean reversion.

    Strategy:
    - RSI period = 14
    - Enter long when RSI crosses below 30
    - Exit long when RSI crosses above 70

    Args:
        data: Dictionary with key 'ohlcv' mapping to a DataFrame that contains a 'close' column.
        params: Dictionary of parameters (not used, kept for compatibility).

    Returns:
        Dict with key 'ohlcv' containing a pd.Series of position targets: 1 for long, 0 for flat.
    """
    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' DataFrame with 'close' column")

    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"].copy()
    if not isinstance(close, pd.Series):
        # If close is a DataFrame column-like, convert to Series
        close = pd.Series(close.squeeze(), index=ohlcv.index)

    period = 14

    # Compute RSI
    rsi = _compute_rsi(close, period=period)

    # Define crossing conditions. Use previous bar to detect crosses.
    prev_rsi = rsi.shift(1)

    # Cross below 30: previous >= 30 and current < 30
    enter_cond = (prev_rsi >= 30) & (rsi < 30)

    # Cross above 70: previous <= 70 and current > 70
    exit_cond = (prev_rsi <= 70) & (rsi > 70)

    # Ensure we don't trigger on NaN values
    valid_mask = rsi.notna() & prev_rsi.notna()
    enter_cond &= valid_mask
    exit_cond &= valid_mask

    # Build position series by iterating through bars to maintain state.
    position = pd.Series(0, index=close.index, dtype=float)
    in_long = False

    # Use integer indexing for speed
    for i in range(len(position)):
        if enter_cond.iat[i] and not in_long:
            in_long = True
            position.iat[i] = 1.0
        elif exit_cond.iat[i] and in_long:
            in_long = False
            position.iat[i] = 0.0
        else:
            position.iat[i] = 1.0 if in_long else 0.0

    return {"ohlcv": position}
