import pandas as pd
import numpy as np
from typing import Dict, Any


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series (float) indexed like `close`.
    """
    # Ensure float series
    close = close.astype(float)

    # Price differences
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via EWM (causal)
    # Use min_periods=period so that initial values before warmup remain NaN
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI formula
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where avg_loss or avg_gain are zero
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    # If both zero -> RSI = 50 (no change)
    mask_gain_zero = avg_gain == 0
    mask_loss_zero = avg_loss == 0

    # Work on a copy to avoid SettingWithCopyWarning
    rsi = rsi.copy()
    rsi.loc[mask_loss_zero & (~mask_gain_zero)] = 100.0
    rsi.loc[mask_gain_zero & (~mask_loss_zero)] = 0.0
    rsi.loc[mask_gain_zero & mask_loss_zero] = 50.0

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

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
    """
    # Accept either a dict with 'ohlcv' or a bare DataFrame for convenience
    if isinstance(data, pd.DataFrame):
        ohlcv = data
    elif isinstance(data, dict):
        if "ohlcv" not in data:
            raise ValueError("`data` must contain 'ohlcv' key with a DataFrame containing 'close' column")
        ohlcv = data["ohlcv"]
    else:
        raise TypeError("`data` must be a dict[str, DataFrame] or a DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Extract and validate parameters (only use declared params)
    if not isinstance(params, dict):
        raise TypeError("`params` must be a dict")

    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")

    # Calculate RSI (causal calculation)
    rsi = _calculate_rsi(close, rsi_period)

    # Boolean masks for over/under thresholds
    below = rsi < oversold
    above = rsi > overbought

    # Detect cross events using previous bar (no lookahead)
    prev_below = below.shift(1, fill_value=False)
    prev_above = above.shift(1, fill_value=False)

    cross_below = (~prev_below) & below  # RSI crossed below oversold
    cross_above = (~prev_above) & above  # RSI crossed above overbought

    # Build position series iteratively to avoid double entries/exits
    pos = np.zeros(len(close), dtype=int)
    prev_pos = 0

    # Iterate over bars
    for i in range(len(close)):
        # Entry: only when currently flat and RSI crossed below oversold
        if prev_pos == 0:
            if bool(cross_below.iloc[i]):
                prev_pos = 1
        # Exit: only when currently long and RSI crossed above overbought
        else:
            if bool(cross_above.iloc[i]):
                prev_pos = 0

        pos[i] = prev_pos

    position = pd.Series(pos, index=close.index, name="position")

    # Ensure dtype is integer and contains only 0 or 1
    position = position.astype(int)

    return {"ohlcv": position}
