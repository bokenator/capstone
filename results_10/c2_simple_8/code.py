"""RSI Mean Reversion Signal Generator

Implements a simple long-only RSI mean reversion strategy:
- RSI period (default 14)
- Enter long when RSI crosses below 30
- Exit when RSI crosses above 70

Exports:
- generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

The returned dict must contain the key 'ohlcv' mapping to a pd.Series of positions
(+1 long, 0 flat). This is what the backtest runner expects.

The implementation uses vectorbt.RSI.run when available; otherwise falls back to a
pandas-based RSI implementation (Wilder's smoothing via ewm).
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

# Try to import vectorbt for RSI calculation; if not available, we still provide
# a pandas fallback.
try:
    import vectorbt as vbt  # type: ignore
    HAS_VBT = True
except Exception:
    vbt = None  # type: ignore
    HAS_VBT = False


def _rsi_pandas(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (exponential moving average).

    Args:
        close: Price series
        window: Lookback period for RSI

    Returns:
        RSI as a pandas Series aligned with `close`.
    """
    close = close.astype(float)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing uses alpha = 1/window with ewm (adjust=False)
    # This produces the same results as the original Wilder's RSI formulation.
    roll_up = up.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # Avoid division by zero
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If roll_down is zero (no losses), RSI should be 100. If roll_up is zero (no gains), RSI should be 0.
    rsi = rsi.fillna(0)
    rsi[(roll_down == 0) & (roll_up > 0)] = 100

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dictionary with at least the key 'ohlcv' mapping to a DataFrame that
              contains a 'close' column.
        params: Dictionary of parameters. Supported keys:
            - 'rsi_period' (int): RSI lookback period (default 14)

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series of positions (+1 long, 0 flat).
    """
    if 'ohlcv' not in data:
        raise ValueError("data must contain an 'ohlcv' DataFrame under key 'ohlcv'.")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame with a 'close' column.")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column.")

    close = ohlcv['close']
    # Ensure series type
    if isinstance(close, pd.DataFrame):
        # If accidentally a DataFrame (multi-col), take the first column
        close = close.iloc[:, 0]

    # Parameters
    rsi_period = int(params.get('rsi_period', 14))

    # Compute RSI using vectorbt if available, otherwise use pandas fallback
    try:
        if HAS_VBT and vbt is not None:
            # vbt.RSI.run returns an object with attribute `.rsi`
            rsi_res = vbt.RSI.run(close, window=rsi_period)
            rsi = rsi_res.rsi
            # If vbt returns a DataFrame with one column, convert to Series
            if isinstance(rsi, pd.DataFrame) and rsi.shape[1] == 1:
                rsi = rsi.iloc[:, 0]
        else:
            raise RuntimeError("vectorbt not available - using fallback RSI implementation")
    except Exception:
        # Fallback to pandas implementation
        rsi = _rsi_pandas(close, window=rsi_period)

    # Ensure RSI is a Series aligned with close
    if isinstance(rsi, pd.DataFrame):
        # Select first column if multiple exist
        rsi = rsi.iloc[:, 0]
    rsi = rsi.reindex(close.index)

    # Detect crossings:
    # Entry: RSI crosses below 30 (previous >= 30, current < 30)
    # Exit:  RSI crosses above 70 (previous <= 70, current > 70)
    prev_rsi = rsi.shift(1)
    entry_signals = (prev_rsi >= 30) & (rsi < 30)
    exit_signals = (prev_rsi <= 70) & (rsi > 70)

    # Replace NaN with False for signal arrays
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    # Build position series using a simple state machine to avoid lookahead
    positions = pd.Series(0, index=close.index, dtype='int8')
    in_position = 0

    # Iterate through index in order
    for i in range(len(close)):
        # If RSI is NaN at this bar, remain flat
        if pd.isna(rsi.iloc[i]):
            in_position = 0
            positions.iloc[i] = 0
            continue

        if in_position == 0:
            if entry_signals.iloc[i]:
                in_position = 1
        else:
            if exit_signals.iloc[i]:
                in_position = 0

        positions.iloc[i] = in_position

    return {'ohlcv': positions}
