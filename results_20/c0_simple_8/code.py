from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Uses an exponential weighted mean with alpha = 1/period (Wilder's smoothing).

    Args:
        close: Price series (pd.Series) indexed by timestamps.
        period: RSI lookback period.

    Returns:
        pd.Series containing RSI values (0-100) aligned with `close` index. Values
        before enough data are available will be NaN.
    """
    # Ensure float dtype for calculations
    close = close.astype(float)

    # Price change
    delta = close.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (exponential with alpha = 1/period)
    # min_periods=period ensures NaN until we have `period` observations
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Keep alignment and NaNs where appropriate
    rsi = rsi.reindex(close.index)
    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position targets for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute RSI (default period 14)
    - Entry (go long): RSI crosses below entry_threshold (default 30)
    - Exit (close long): RSI crosses above exit_threshold (default 70)

    The function returns a dict with a single key 'ohlcv' mapping to a pd.Series of
    position targets: 1 = long, 0 = flat. The series is aligned to the input
    `data['ohlcv']['close']` index.

    Args:
        data: Dict containing at least the key 'ohlcv' mapped to a DataFrame with a
              'close' column. Example: data['ohlcv']['close'] -> pd.Series
        params: Dict of parameters. Supported keys:
            - 'rsi_period' (int): lookback period for RSI (default 14)
            - 'entry_threshold' (float): RSI threshold to enter long (default 30)
            - 'exit_threshold' (float): RSI threshold to exit long (default 70)

    Returns:
        Dict[str, pd.Series]: {'ohlcv': position_series}

    Raises:
        ValueError: If required data is missing or malformed.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict containing an 'ohlcv' DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("`data` must contain 'ohlcv' key with a DataFrame value")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    # Attempt to get the close series (support common column name variants)
    close_col_candidates = ['close', 'Close', 'adjclose', 'adj_close', 'Adj Close']
    close_series = None
    for col in close_col_candidates:
        if col in ohlcv.columns:
            close_series = ohlcv[col].astype(float)
            break

    if close_series is None:
        # As a fallback, if there's only one column assume it's the close
        if ohlcv.shape[1] == 1:
            close_series = ohlcv.iloc[:, 0].astype(float)
        else:
            raise ValueError("Could not find a 'close' column in data['ohlcv']")

    # Parameters with defaults
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14
    entry_threshold = float(params.get('entry_threshold', 30.0)) if params is not None else 30.0
    exit_threshold = float(params.get('exit_threshold', 70.0)) if params is not None else 70.0

    if rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    # Compute RSI
    rsi = _compute_rsi(close_series, period=rsi_period)

    # If RSI is completely NaN (not enough data), return flat positions
    if rsi.isna().all():
        empty_pos = pd.Series(0, index=close_series.index, dtype='int8')
        return {'ohlcv': empty_pos}

    # Define crossing conditions. Use shift to detect cross events. Fill NaN with False
    prev_rsi = rsi.shift(1)
    entry_cond = (rsi < entry_threshold) & (prev_rsi >= entry_threshold)
    exit_cond = (rsi > exit_threshold) & (prev_rsi <= exit_threshold)

    entry_cond = entry_cond.fillna(False)
    exit_cond = exit_cond.fillna(False)

    # Build position series by walking through the index to ensure proper stateful behaviour
    position = pd.Series(0, index=close_series.index, dtype='int8')
    in_position = False

    # Iterate over aligned boolean arrays for speed
    for ts, enter, exit_ in zip(close_series.index, entry_cond.values, exit_cond.values):
        if not in_position:
            if enter:
                in_position = True
                position.loc[ts] = 1
            else:
                position.loc[ts] = 0
        else:
            # Currently long
            if exit_:
                in_position = False
                position.loc[ts] = 0
            else:
                position.loc[ts] = 1

    # Ensure dtype is numeric (0/1) and aligned with input close
    position = position.astype(int)

    return {'ohlcv': position}
