from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Price series.
        period: Lookback period for RSI (default 14).

    Returns:
        RSI as a pandas Series aligned with `close` index. The first `period` bars are set to NaN.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Price changes
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's EMA smoothing. Using min_periods=period to avoid unstable early values.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Relative strength
    rs = avg_gain / avg_loss

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Force warmup NaNs for the initial period to be conservative
    rsi.iloc[:period] = np.nan

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean-reversion strategy.

    Strategy logic:
    - Compute RSI with period (default 14)
    - Go long when RSI crosses below 30 (oversold)
    - Exit when RSI crosses above 70 (overbought)
    - Long-only (positions are 0 or 1)

    Args:
        data: Dictionary expected to contain key 'ohlcv' with a DataFrame that includes a 'close' column.
        params: Parameters dictionary. Supported keys:
            - 'rsi_period' (int): RSI lookback period. Defaults to 14.

    Returns:
        A dictionary with key 'ohlcv' mapping to a pandas Series of positions (0 or 1) aligned to the input 'close' index.

    Raises:
        ValueError: If required data keys/columns are missing.
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing an 'ohlcv' DataFrame")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame that includes a 'close' column")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters
    rsi_period = int(params.get('rsi_period', 14)) if params is not None else 14

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crosses
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below 30 (from >=30 to <30)
    entries = (prev_rsi >= 30.0) & (rsi < 30.0) & prev_rsi.notna() & rsi.notna()

    # Exit: RSI crosses above 70 (from <=70 to >70)
    exits = (prev_rsi <= 70.0) & (rsi > 70.0) & prev_rsi.notna() & rsi.notna()

    # Build position series: long (1) after the last entry until the next exit
    # Use cumulative counts to determine whether we're currently long
    entry_counts = entries.cumsum()
    exit_counts = exits.cumsum()

    position_bool = entry_counts > exit_counts

    position = position_bool.astype(int)

    # Ensure the returned Series aligns with the input close index and has no unexpected name
    position = pd.Series(position, index=close.index, name='position')

    # Fill any remaining NaNs with 0 (flat)
    position = position.fillna(0).astype(int)

    return {'ohlcv': position}
