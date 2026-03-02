import numpy as np
import pandas as pd
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    This implementation is causal (no lookahead): uses only past and current prices.

    Args:
        close: Price series.
        period: RSI lookback period.

    Returns:
        RSI series (0-100), indexed like `close`.
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing -> use ewm with alpha=1/period and adjust=False for recursive calculation
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)

    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is zero -> RSI should be 100 (all gains)
    rsi = rsi.fillna(0)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    # When avg_gain is zero -> RSI should be 0 (all losses)
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position signals based on RSI mean reversion.

    Strategy logic:
    - Compute RSI with period params['rsi_period']
    - Enter long when RSI crosses below params['oversold']
    - Exit long when RSI crosses above params['overbought']

    Args:
        data: Dictionary containing 'ohlcv' -> DataFrame with 'close' column.
        params: Dictionary with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed like input close.
    """
    # Validate inputs
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")

    ohlcv = data['ohlcv']
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].copy()

    # Read params with defaults
    rsi_period = int(params.get('rsi_period', 14))
    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    if rsi_period <= 0:
        raise ValueError('rsi_period must be > 0')

    # Compute RSI (causal)
    rsi = _compute_rsi(close, period=rsi_period)

    # Detect crossings (use previous bar vs current bar)
    prev_rsi = rsi.shift(1)

    # Entry: RSI crosses below oversold (prev > oversold and curr <= oversold)
    entries_raw = (prev_rsi > oversold) & (rsi <= oversold)

    # Exit: RSI crosses above overbought (prev < overbought and curr >= overbought)
    exits_raw = (prev_rsi < overbought) & (rsi >= overbought)

    # Build position series (0 flat, 1 long) using a simple state machine to avoid double entries/exits
    idx = close.index
    n = len(idx)

    entries_arr = entries_raw.reindex(idx, fill_value=False).to_numpy(dtype=bool)
    exits_arr = exits_raw.reindex(idx, fill_value=False).to_numpy(dtype=bool)

    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        if entries_arr[i] and state == 0:
            state = 1
        elif exits_arr[i] and state == 1:
            state = 0
        pos[i] = state

    position_series = pd.Series(pos, index=idx, name='position')

    # Ensure dtype is integer and contains only 0 or 1
    position_series = position_series.astype(int)

    return {"ohlcv": position_series}
