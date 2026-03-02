import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing (EWMA with alpha=1/period).
    This implementation is causal (no lookahead) and returns a Series aligned with the input index.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        RSI lookback period.

    Returns
    -------
    pd.Series
        RSI values (0-100) with NaN for the initial periods until enough data is available.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via EWM with adjust=False is causal
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Prevent division warnings
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss == 0
    # If avg_loss == 0 and avg_gain > 0 -> RSI = 100
    mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    if mask_loss_zero.any():
        rsi.loc[mask_loss_zero] = 100.0

    # If avg_gain == 0 and avg_loss > 0 -> RSI = 0
    mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)
    if mask_gain_zero.any():
        rsi.loc[mask_gain_zero] = 0.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean-reversion strategy.

    Strategy logic:
    - Calculate RSI with period = params['rsi_period'] (default 14)
    - Go long when RSI crosses below params['oversold'] (default 30)
    - Exit when RSI crosses above params['overbought'] (default 70)

    Returns a dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), indexed the same as input close prices.

    Parameters
    ----------
    data : dict
        data["ohlcv"] must be a pd.DataFrame containing a 'close' column.
    params : dict
        Contains 'rsi_period', 'oversold', 'overbought'.

    Returns
    -------
    dict
        {"ohlcv": position_series}
    """
    # Validate input
    if not isinstance(data, dict) or "ohlcv" not in data:
        raise ValueError("data must be a dict with key 'ohlcv' containing a DataFrame with a 'close' column")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must be a DataFrame containing a 'close' column")

    close = ohlcv["close"].astype(float)

    # Read parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period <= 0:
        raise ValueError("rsi_period must be > 0")

    # Compute RSI (causal)
    rsi = compute_rsi(close, rsi_period)

    # Define cross events using previous value -> no lookahead
    prev_rsi = rsi.shift(1)

    # Entry: prev >= oversold and current < oversold (crosses below)
    entry_signals = (prev_rsi >= oversold) & (rsi < oversold)

    # Exit: prev <= overbought and current > overbought (crosses above)
    exit_signals = (prev_rsi <= overbought) & (rsi > overbought)

    # Convert to boolean numpy arrays (NaNs -> False) for fast loop and to avoid lookahead
    entry_arr = entry_signals.fillna(False).to_numpy(dtype=bool)
    exit_arr = exit_signals.fillna(False).to_numpy(dtype=bool)

    n = len(close)
    pos_arr = np.zeros(n, dtype=np.int8)

    current_pos = 0  # 0 = flat, 1 = long
    for i in range(n):
        # Entry only if not already long
        if entry_arr[i] and current_pos == 0:
            current_pos = 1
        # Exit only if currently long
        elif exit_arr[i] and current_pos == 1:
            current_pos = 0
        pos_arr[i] = current_pos

    position = pd.Series(pos_arr, index=close.index, name="position").astype(int)

    # Ensure no NaNs (positions should be 0 or 1)
    position = position.fillna(0).astype(int)

    return {"ohlcv": position}
