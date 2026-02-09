from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI period (int).

    Returns:
        RSI series with the same index as close. Initial values (warmup) will be NaN.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    # Ensure numeric
    close_numeric = pd.to_numeric(close, errors="coerce")

    # Price changes
    delta = close_numeric.diff()

    # Gains and losses
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm with alpha=1/period
    # min_periods=period ensures we don't produce values before enough data
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 => RSI = 0
    # - If both are zero => RSI = 50 (no price movement)
    mask_loss_zero = (avg_loss == 0) & (avg_gain.notna())
    mask_gain_zero = (avg_gain == 0) & (avg_loss.notna())

    both_zero = mask_loss_zero & mask_gain_zero
    only_loss_zero = mask_loss_zero & ~mask_gain_zero
    only_gain_zero = mask_gain_zero & ~mask_loss_zero

    rsi = rsi.copy()
    rsi[only_loss_zero] = 100.0
    rsi[only_gain_zero] = 0.0
    rsi[both_zero] = 50.0

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Strategy:
    - Compute RSI with period = params['rsi_period']
    - Enter long when RSI crosses below params['oversold']
    - Exit long when RSI crosses above params['overbought']

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' with 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to pd.Series of positions (1 for long, 0 for flat).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with a DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv["close"]

    # Validate and read params
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    # Only use declared params
    try:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
    except Exception as e:
        raise ValueError(f"Invalid parameter types: {e}")

    # Bound checks according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if overbought <= oversold:
        raise ValueError("overbought must be greater than oversold")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Entry: RSI crosses below oversold (previous bar >= oversold, current < oversold)
    prev_rsi = rsi.shift(1)
    entry_cond = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (previous bar <= overbought, current > overbought)
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaN conditions with False to avoid spurious signals during warmup
    entry_cond = entry_cond.fillna(False)
    exit_cond = exit_cond.fillna(False)

    n = len(close)
    positions = np.zeros(n, dtype=int)

    in_position = False
    # Iterate to build position series in a stateful manner
    for i in range(n):
        if in_position:
            # If currently long, check for exit first
            if bool(exit_cond.iat[i]):
                in_position = False
                positions[i] = 0
            else:
                positions[i] = 1
        else:
            # Not in position, check for entry
            if bool(entry_cond.iat[i]):
                in_position = True
                positions[i] = 1
            else:
                positions[i] = 0

    position_series = pd.Series(positions, index=close.index).astype(int)

    return {"ohlcv": position_series}
