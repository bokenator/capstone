import numpy as np
import pandas as pd
from typing import Dict, Any


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using simple moving averages.

    This implementation is backward-looking (no lookahead). It uses a rolling
    window of previous `period` returns to compute average gains and losses
    (SMA-based RSI). The function handles edge cases where average loss is
    zero to avoid division by zero.

    Args:
        close: Price series.
        period: RSI lookback period (integer >= 2).

    Returns:
        RSI series with the same index as `close`. Values will be NaN for
        the first (period - 1) entries where insufficient data is available.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    close = close.astype(float)
    delta = close.diff()

    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    # Simple moving averages of gains and losses (backward-looking)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    # Compute RS and RSI safely
    # Replace zeros in avg_loss with NaN for the division to avoid inf
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly
    # If avg_loss == 0 and avg_gain > 0 => RSI = 100
    mask_loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
    rsi[mask_loss_zero_gain_pos] = 100.0

    # If avg_gain == 0 and avg_loss > 0 => RSI = 0
    mask_gain_zero_loss_pos = (avg_gain == 0) & (avg_loss > 0)
    rsi[mask_gain_zero_loss_pos] = 0.0

    # If both are zero (no price movement) => RSI = 50 (neutral)
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi[mask_both_zero] = 50.0

    return rsi


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Strategy logic:
    - Compute RSI with period = params['rsi_period']
    - Go long when RSI crosses below params['oversold'] from above (0->1)
    - Exit (go flat) when RSI crosses above params['overbought'] from below (1->0)
    - Long-only, single asset. Position series contains 0 (flat) or 1 (long).

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with a DataFrame that has a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1),
        indexed the same as input close prices.
    """
    # Validate input structure
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key")
    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain a 'close' column")

    # Extract parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Basic validation of parameters
    if rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")

    close = ohlcv['close'].copy()

    # Compute RSI (backward-looking)
    rsi = _compute_rsi(close, rsi_period)

    # Prepare previous RSI for crossing detection
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (was >= oversold, now < oversold)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: RSI crosses above overbought (was <= overbought, now > overbought)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Initialize position series (0 = flat, 1 = long)
    position = pd.Series(0, index=close.index, dtype='int8')

    # Iterate to enforce single-entry / single-exit behavior without lookahead
    in_position = 0
    for idx in range(len(close)):
        if in_position == 0:
            # Can only enter if entry signal at this bar
            if bool(entry_signals.iloc[idx]):
                in_position = 1
        else:
            # Currently long; can only exit if exit signal at this bar
            if bool(exit_signals.iloc[idx]):
                in_position = 0
        position.iloc[idx] = in_position

    # Ensure positions contain only 0 or 1 and align with input index
    position = position.astype('int8')

    return {'ohlcv': position}
