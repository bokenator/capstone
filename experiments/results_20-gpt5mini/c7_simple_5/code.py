import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames or a single DataFrame. Must contain
              'ohlcv' key with DataFrame having 'close' column, or be a DataFrame itself.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
    """

    # --- Input validation and normalization ---
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        if 'ohlcv' not in data:
            raise KeyError("Input data dict must contain 'ohlcv' key with a DataFrame.")
        df = data['ohlcv']
    else:
        raise TypeError("data must be a pandas DataFrame or a dict with key 'ohlcv'.")

    if 'close' not in df.columns:
        raise KeyError("Input DataFrame must contain 'close' column as per DATA_SCHEMA.")

    # Extract close prices (ensure float dtype)
    close = df['close'].astype(float).copy()

    # Parameters with basic validation
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period < 2:
        raise ValueError('rsi_period must be >= 2')

    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))
    if not (0.0 <= oversold <= 100.0 and 0.0 <= overbought <= 100.0):
        raise ValueError('oversold and overbought must be between 0 and 100')
    if oversold >= overbought:
        raise ValueError('oversold threshold must be less than overbought threshold')

    # --- RSI calculation (Wilder's smoothing using ewm) ---
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing: ewm with alpha = 1/period and adjust=False
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Compute RSI
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle edge cases explicitly
        rsi = rsi.astype(float)
        # If avg_gain and avg_loss are both zero -> set neutral 50
        mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi.loc[mask_both_zero] = 50.0
        # If avg_loss is zero (and avg_gain > 0) -> RSI = 100
        mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
        rsi.loc[mask_loss_zero] = 100.0
        # If avg_gain is zero (and avg_loss > 0) -> RSI = 0
        mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)
        rsi.loc[mask_gain_zero] = 0.0

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # --- Generate position series (0 or 1) using crossing logic ---
    n = len(close)
    positions = np.zeros(n, dtype=int)

    # Iterate through time ensuring decisions only use present and past data (no lookahead)
    for i in range(1, n):
        prev_pos = positions[i - 1]

        # If RSI not available at current or previous index, carry forward position
        if pd.isna(rsi.iat[i]) or pd.isna(rsi.iat[i - 1]):
            positions[i] = prev_pos
            continue

        prev_rsi = float(rsi.iat[i - 1])
        curr_rsi = float(rsi.iat[i])

        if prev_pos == 0:
            # Enter long when RSI crosses BELOW oversold: prev >= oversold and curr < oversold
            if (prev_rsi >= oversold) and (curr_rsi < oversold):
                positions[i] = 1
            else:
                positions[i] = 0
        else:
            # Exit when RSI crosses ABOVE overbought: prev <= overbought and curr > overbought
            if (prev_rsi <= overbought) and (curr_rsi > overbought):
                positions[i] = 0
            else:
                positions[i] = 1

    # Ensure the returned Series has same index and no NaNs (positions are integers 0/1)
    position_series = pd.Series(positions, index=close.index, name='position').astype(int)

    return {'ohlcv': position_series}
