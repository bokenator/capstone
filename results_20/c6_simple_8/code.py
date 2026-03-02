# RSI mean reversion signal generator
# Implements: RSI(period=14), go long when RSI crosses below oversold (default 30), exit when RSI crosses above overbought (default 70)

from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_signals(data: dict, params: dict) -> dict:
    """Generate long-only position series based on RSI mean reversion.

    Args:
        data (dict): Dictionary containing 'ohlcv' DataFrame with a 'close' column.
        params (dict): Parameters with keys:
            - 'rsi_period' (int): Period for RSI calculation (default 14)
            - 'oversold' (float): Oversold threshold (default 30.0)
            - 'overbought' (float): Overbought threshold (default 70.0)

    Returns:
        dict: {"ohlcv": position_series} where position_series is a pd.Series of 0 (flat) or 1 (long)
    """
    # Validate inputs
    if not isinstance(data, dict) or 'ohlcv' not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].astype(float).copy()

    # Read params with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Calculate RSI using Wilder's smoothing (EWMA with alpha=1/period)
    # Delta
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential weighted mean with alpha=1/period (Wilder's smoothing)
    # Use adjust=False to make it recursive (uses only past values)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RS and RSI
    rs = avg_gain / avg_loss
    # Handle division by zero: when avg_loss == 0 -> RSI = 100
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    # If both avg_gain and avg_loss are zero, define RSI as 50 (neutral)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    # Ensure rsi is a pandas Series aligned with close index
    rsi = pd.Series(rsi, index=close.index, name='rsi')
    rsi.loc[both_zero] = 50.0

    # Prepare previous RSI for crossing detection
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below oversold (or is below on first valid bar)
    entry_cond = (rsi < oversold) & ((prev_rsi >= oversold) | (prev_rsi.isna()))

    # Exit when RSI crosses above overbought (or is above on first valid bar)
    exit_cond = (rsi > overbought) & ((prev_rsi <= overbought) | (prev_rsi.isna()))

    # State machine to generate position series (0 or 1)
    position = pd.Series(0, index=close.index, dtype='int8')

    is_long = False
    # Use integer-based index iteration for deterministic behavior
    for i in range(len(close)):
        if not is_long:
            # If entry condition is true at this bar -> enter long
            try:
                if bool(entry_cond.iloc[i]):
                    is_long = True
            except Exception:
                # In case of NaN/invalid boolean, do nothing
                pass
        else:
            # If currently long and exit condition is true -> exit
            try:
                if bool(exit_cond.iloc[i]):
                    is_long = False
            except Exception:
                pass

        position.iloc[i] = 1 if is_long else 0

    # Return dict with same key 'ohlcv' as required by the harness
    return {'ohlcv': position}
