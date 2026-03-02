from typing import Dict, Any

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        pd.Series with RSI values (NaN for the initial warmup period < period).
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    close = close.astype(float)
    delta = close.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (exponential moving average with alpha=1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Relative strength and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle cases with zero loss -> RSI = 100
    rsi = rsi.fillna(0)
    zero_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi[zero_loss_mask] = 100.0

    # If both avg_gain and avg_loss are zero (flat price), set RSI to 50
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi[flat_mask] = 50.0

    # Preserve NaNs for the warmup period explicitly: set first (period-1) to NaN
    if len(rsi) >= period:
        rsi.iloc[: period] = np.nan

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean reversion.

    Strategy logic:
    - Calculate RSI with period params['rsi_period']
    - Go long when RSI crosses below params['oversold'] (from above to below)
    - Exit (go flat) when RSI crosses above params['overbought'] (from below to above)

    Args:
        data: Dict with key "ohlcv" -> DataFrame containing at least a 'close' column.
        params: Dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict{"ohlcv": pd.Series} where the series contains 0 (flat) or 1 (long) values.
    """
    # Validate inputs
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be > 0")

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions using only past information (shift(1))
    prev_rsi = rsi.shift(1)

    # Entry when RSI crosses below oversold: prev >= oversold and curr < oversold
    entry_cross = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit when RSI crosses above overbought: prev <= overbought and curr > overbought
    exit_cross = (prev_rsi <= overbought) & (rsi > overbought)

    # Prepare position series (0 or 1) and iterate to enforce single-entry semantics
    position = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate over positions by index to ensure statefulness without lookahead
    for i in range(len(close)):
        if not in_position:
            # Only enter if an entry_cross occurs
            if bool(entry_cross.iloc[i]):
                in_position = True
                position.iloc[i] = 1
            else:
                position.iloc[i] = 0
        else:
            # If currently long, exit only on an exit_cross
            if bool(exit_cross.iloc[i]):
                in_position = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1

    # Ensure there are no NaNs in the output (fill any remaining with 0)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
