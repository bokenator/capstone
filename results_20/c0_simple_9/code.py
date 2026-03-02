from typing import Any
import pandas as pd
import numpy as np
import vectorbt as vbt


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series (float), aligned with close index. Values may contain NaN for warmup bars.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Ensure float dtype for computations
    close = close.astype(float)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (RMA) implemented via EWM with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Initialize RSI series with NaNs
    rsi = pd.Series(index=close.index, dtype=float)

    # Valid mask where we have both averages available
    valid_mask = (~avg_gain.isna()) & (~avg_loss.isna())

    # Normal case where avg_loss != 0
    normal_mask = valid_mask & (avg_loss != 0)
    rsi.loc[normal_mask] = 100.0 - (100.0 / (1.0 + (avg_gain.loc[normal_mask] / avg_loss.loc[normal_mask])))

    # avg_loss == 0 and avg_gain > 0 -> RSI = 100
    mask_loss_zero = valid_mask & (avg_loss == 0) & (avg_gain > 0)
    rsi.loc[mask_loss_zero] = 100.0

    # avg_gain == 0 and avg_loss > 0 -> RSI = 0
    mask_gain_zero = valid_mask & (avg_gain == 0) & (avg_loss > 0)
    rsi.loc[mask_gain_zero] = 0.0

    # both zero -> no price movement -> RSI = 50 (neutral)
    mask_both_zero = valid_mask & (avg_gain == 0) & (avg_loss == 0)
    rsi.loc[mask_both_zero] = 50.0

    return rsi


def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Strategy logic:
    - RSI period = 14 (default)
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single asset

    Args:
        data: Dict containing at least 'ohlcv' -> DataFrame with 'close' column.
        params: Dict of parameters. Supported keys:
            - 'rsi_period' (int, optional): RSI lookback period (default 14)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of position targets (1 for long, 0 for flat).
    """
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping names to DataFrames")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].astype(float)

    # Parameters
    rsi_period = int(params.get("rsi_period", 14)) if params is not None else 14
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Prepare crossing signals
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below 30 (previous >= 30, current < 30)
    entries = (prev_rsi >= 30.0) & (rsi < 30.0)

    # Exit: RSI crosses above 70 (previous <= 70, current > 70)
    exits = (prev_rsi <= 70.0) & (rsi > 70.0)

    # Build position series (1 for long, 0 for flat)
    positions = np.zeros(len(rsi), dtype=int)
    in_position = False

    entries_vals = entries.fillna(False).values
    exits_vals = exits.fillna(False).values

    for i in range(len(positions)):
        if entries_vals[i] and not in_position:
            in_position = True
            positions[i] = 1
        elif exits_vals[i] and in_position:
            in_position = False
            positions[i] = 0
        else:
            positions[i] = 1 if in_position else 0

    position_series = pd.Series(positions, index=close.index, name="position")

    return {"ohlcv": position_series}
