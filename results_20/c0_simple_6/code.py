import pandas as pd
import numpy as np
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    This implementation uses an exponential weighted mean with alpha=1/period
    (Wilder's smoothing). It handles edge cases where the average loss is
    zero.

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series aligned with the input index.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing via EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative Strength
    # Avoid division by zero: where avg_loss == 0, set RSI to 100 if avg_gain > 0,
    # to 50 if avg_gain == 0 (no movement), otherwise compute normally.
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle avg_loss == 0 cases explicitly
    loss_zero = avg_loss == 0
    if loss_zero.any():
        rsi = rsi.copy()
        # avg_gain == 0 => no movement => RSI = 50
        rsi[loss_zero & (avg_gain == 0)] = 50.0
        # avg_gain > 0 and avg_loss == 0 => RSI = 100
        rsi[loss_zero & (avg_gain > 0)] = 100.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy logic:
    - Compute RSI with period (default 14)
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only, single asset. Position series values: 1 = long, 0 = flat.

    Args:
        data: Dictionary of dataframes. Expects data['ohlcv'] to be a DataFrame
              containing a 'close' column (pd.Series).
        params: Dictionary of parameters. Supported keys (optional):
            - 'rsi_period' (int): RSI lookback period. Default 14.
            - 'rsi_oversold' (float): Oversold threshold. Default 30.
            - 'rsi_overbought' (float): Overbought threshold. Default 70.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions aligned with
        the input close series index.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing an 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close_series = ohlcv['close']
    # If close column is a DataFrame (e.g., multi-column), pick the first column
    if isinstance(close_series, pd.DataFrame):
        if close_series.shape[1] == 0:
            raise ValueError("'close' column has no data")
        close = close_series.iloc[:, 0]
    else:
        close = close_series

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    oversold = float(params.get('rsi_oversold', 30.0)) if params is not None else 30.0
    overbought = float(params.get('rsi_overbought', 70.0)) if params is not None else 70.0

    if rsi_period < 1:
        raise ValueError('rsi_period must be >= 1')

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crosses: entry when RSI crosses below oversold, exit when crosses above overbought
    prev_rsi = rsi.shift(1)

    enter_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs from warmup with False so they don't trigger trades
    enter_signal = enter_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # Build position series by iterating through time (stateful)
    position_array = np.zeros(len(close), dtype=int)
    in_position = False

    # Use positional indexing for speed and to avoid alignment surprises
    enter_vals = enter_signal.values
    exit_vals = exit_signal.values

    for i in range(len(position_array)):
        if not in_position and enter_vals[i]:
            in_position = True
            position_array[i] = 1
        elif in_position and exit_vals[i]:
            in_position = False
            position_array[i] = 0
        elif in_position:
            position_array[i] = 1
        else:
            position_array[i] = 0

    positions = pd.Series(position_array, index=close.index, name='position')

    return {'ohlcv': positions}
