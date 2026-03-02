from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EMA with alpha=1/period).

    Args:
        series: Price series (typically close prices).
        period: RSI lookback period.

    Returns:
        RSI as a pandas Series aligned with the input series.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    # Price differences
    delta = series.diff()

    # Gains and losses
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing (RMA) via EWM with alpha=1/period
    # adjust=False ensures recursive (Wilder) smoothing
    avg_gain = up.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss

    # RSI calculation with handling for avg_loss == 0
    rsi = pd.Series(np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs)), index=series.index)

    # Keep NaNs where they are expected (start of series)
    rsi = rsi.where(~series.isna())

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean-reversion strategy.

    Strategy logic:
    - Compute 14-period RSI
    - Enter long when RSI crosses below 30 (prev_rsi > 30 and rsi <= 30)
    - Exit long when RSI crosses above 70 (prev_rsi < 70 and rsi >= 70)
    - Long-only, single asset

    Args:
        data: Dict containing at least the key 'ohlcv' mapped to a DataFrame with a 'close' column.
        params: Parameters dict (not required for this implementation). The RSI period can be
                overridden via params['rsi_period'] if desired.

    Returns:
        Dict with key 'ohlcv' mapping to a pandas Series of position targets (0 or 1), indexed
        the same as the input close prices.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing an 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame value")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv['close'].astype(float)

    # Parameters
    rsi_period = int(params.get('rsi_period', 14)) if isinstance(params, dict) else 14

    # Compute RSI
    rsi = _compute_rsi(close, period=rsi_period)

    # Previous RSI for crossover detection
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below 30
    entry_mask = (prev_rsi > 30) & (rsi <= 30)

    # Exit: RSI crosses above 70
    exit_mask = (prev_rsi < 70) & (rsi >= 70)

    # Ensure boolean dtype and align index
    entry_mask = entry_mask.fillna(False).astype(bool)
    exit_mask = exit_mask.fillna(False).astype(bool)

    # Build position series (0 or 1) by iterating through timestamps to respect stateful entries/exits
    positions = pd.Series(0, index=close.index, dtype=int)

    in_position = False
    # Iterate in order; using index ensures we correctly align with timestamps
    for idx in positions.index:
        if not in_position:
            # If not in position, check entry
            if entry_mask.loc[idx]:
                in_position = True
        else:
            # If in position, check exit
            if exit_mask.loc[idx]:
                in_position = False

        positions.loc[idx] = 1 if in_position else 0

    # Return signals keyed by 'ohlcv' to match runner expectations
    return {'ohlcv': positions}
