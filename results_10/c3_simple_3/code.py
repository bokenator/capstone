import pandas as pd
import numpy as np
from typing import Any, Dict


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).

    This implementation is causal (uses only past data) and will produce NaN
    for the initial warm-up period when using min_periods=period.

    Args:
        close: Series of close prices.
        period: RSI lookback period.

    Returns:
        pd.Series of RSI values (same index as close).
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via ewm with alpha=1/period and adjust=False is causal
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly
    mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
    mask_loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
    mask_gain_zero_loss_pos = (avg_gain == 0) & (avg_loss > 0)

    # Use .loc to avoid SettingWithCopy warnings
    rsi = rsi.copy()
    if mask_both_zero.any():
        rsi.loc[mask_both_zero] = 50.0
    if mask_loss_zero_gain_pos.any():
        rsi.loc[mask_loss_zero_gain_pos] = 100.0
    if mask_gain_zero_loss_pos.any():
        rsi.loc[mask_gain_zero_loss_pos] = 0.0

    return rsi


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate long-only position targets (0 or 1) based on RSI mean-reversion.

    Strategy logic:
    - Compute RSI (period = params['rsi_period']).
    - Go long (position=1) when RSI crosses below params['oversold'].
    - Exit to flat (position=0) when RSI crosses above params['overbought'].

    The function accepts either:
    - data: dict with key 'ohlcv' containing a DataFrame with a 'close' column, OR
    - data: a DataFrame itself (for compatibility with some test harnesses).

    Returns a dict {'ohlcv': position_series} where position_series is a pd.Series
    indexed the same as the input close prices and contains only 0 or 1.

    Args:
        data: Input market data (dict or DataFrame).
        params: Parameters containing 'rsi_period', 'oversold', 'overbought'.

    Returns:
        Dict with key 'ohlcv' mapping to the position pd.Series.
    """
    # Accept either dict with 'ohlcv' or DataFrame directly
    if isinstance(data, pd.DataFrame):
        ohlcv = data.copy()
    elif isinstance(data, dict) and "ohlcv" in data and isinstance(data["ohlcv"], pd.DataFrame):
        ohlcv = data["ohlcv"].copy()
    else:
        raise ValueError("data must be a DataFrame or a dict with key 'ohlcv' containing a DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain a 'close' column")

    close = ohlcv["close"].astype(float).copy()

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    # Fill forward NaNs in price (uses only past information). Do not bfill to avoid lookahead.
    close_filled = close.fillna(method="ffill")

    # Compute RSI (causal)
    rsi = _compute_rsi(close_filled, rsi_period)

    # Prepare position series (0 = flat, 1 = long)
    positions = pd.Series(0, index=close.index, dtype="int64")

    state = 0  # 0 = flat, 1 = long

    # Iterate and apply entry/exit rules based on RSI cross
    for i in range(len(close)):
        rsi_curr = rsi.iloc[i]
        rsi_prev = rsi.iloc[i - 1] if i > 0 else np.nan

        if state == 0:
            # Entry: RSI crosses below oversold
            # Allow entry if previous is NaN but current is below threshold
            can_enter = False
            if not pd.isna(rsi_curr):
                if pd.isna(rsi_prev):
                    can_enter = rsi_curr < oversold
                else:
                    can_enter = (rsi_prev >= oversold) and (rsi_curr < oversold)

            if can_enter:
                state = 1

        else:
            # Exit: RSI crosses above overbought
            can_exit = False
            if not pd.isna(rsi_curr):
                if pd.isna(rsi_prev):
                    can_exit = rsi_curr > overbought
                else:
                    can_exit = (rsi_prev <= overbought) and (rsi_curr > overbought)

            if can_exit:
                state = 0

        positions.iloc[i] = state

    # Ensure only 0 or 1 values and no NaNs
    positions = positions.fillna(0).astype(int)

    return {"ohlcv": positions}
