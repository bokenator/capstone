from typing import Dict, Any

import numpy as np
import pandas as pd

# Optional import of vectorbt. We verify its API via documentation search before using it.
try:
    import vectorbt as vbt
    HAS_VBT = True
except Exception:
    vbt = None
    HAS_VBT = False


def _compute_rsi_with_vbt(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using vectorbt's RSI indicator.

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        pd.Series of RSI values aligned with `close` index.
    """
    # vbt.RSI.run returns an indicator result with attribute `.rsi`.
    # If input is a Series, `.rsi` may be a Series or a single-column DataFrame.
    res = vbt.RSI.run(close, window=period)
    rsi = res.rsi
    if isinstance(rsi, pd.DataFrame):
        # Select the first column if a DataFrame is returned
        rsi = rsi.iloc[:, 0]
    # Ensure index alignment
    rsi = rsi.reindex(close.index)
    return rsi


def _compute_rsi_fallback(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using a fallback implementation (Wilder's smoothing via EWM).
    This ensures the function works even if vectorbt is not available.

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        pd.Series of RSI values aligned with `close` index.
    """
    close = close.astype(float)
    # Standard RSI calculation
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing approximation using EWM with alpha=1/period
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero; compute RS then RSI
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Replace inf values resulting from division by zero
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    # Logical corrections:
    # - If roll_down == 0 and roll_up > 0 => RSI = 100
    # - If roll_up == 0 and roll_down > 0 => RSI = 0
    # - If both are zero => RSI = 50 (no change)
    cond_down_zero = roll_down == 0
    cond_up_zero = roll_up == 0

    rsi = rsi.copy()
    rsi[cond_down_zero & (roll_up > 0)] = 100.0
    rsi[cond_up_zero & (roll_down > 0)] = 0.0
    rsi[cond_up_zero & cond_down_zero] = 50.0

    # Align index
    rsi = rsi.reindex(close.index)
    return rsi


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using vectorbt if available, otherwise use fallback.
    """
    if HAS_VBT and vbt is not None:
        try:
            return _compute_rsi_with_vbt(close, period)
        except Exception:
            # Fall back if vectorbt fails for any reason
            return _compute_rsi_fallback(close, period)
    else:
        return _compute_rsi_fallback(close, period)


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate long-only position signals based on RSI mean-reversion rule.

    Strategy Logic:
    - Calculate RSI with period = params['rsi_period'] (default 14)
    - Go long (position=1) when RSI crosses below params['oversold'] (default 30)
    - Exit to flat (position=0) when RSI crosses above params['overbought'] (default 70)

    Args:
        data: dict with key 'ohlcv' containing a DataFrame with a 'close' column.
        params: dict with keys 'rsi_period', 'oversold', 'overbought'.

    Returns:
        dict with key 'ohlcv' mapping to a pd.Series of positions (0 or 1), same index as input.
    """
    # Validate input
    if not isinstance(data, dict):
        raise TypeError("data must be a dict containing 'ohlcv' DataFrame")
    if 'ohlcv' not in data:
        raise KeyError("data must contain 'ohlcv' key")

    ohlcv = data['ohlcv']
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv['close'].copy()
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Params with defaults
    rsi_period = int(params.get('rsi_period', 14))
    if rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")

    oversold = float(params.get('oversold', 30.0))
    overbought = float(params.get('overbought', 70.0))

    # Compute RSI
    rsi = _compute_rsi(close, rsi_period)

    # Build position series iteratively to avoid double entries and ensure no lookahead
    n = len(close)
    positions = pd.Series(0, index=close.index, dtype=int)

    pos = 0  # current position: 0 = flat, 1 = long

    for i in range(n):
        curr_rsi = rsi.iloc[i]
        prev_rsi = rsi.iloc[i - 1] if i > 0 else np.nan

        # If RSI is NaN at current bar, maintain current position
        if pd.isna(curr_rsi):
            positions.iloc[i] = pos
            continue

        if pos == 0:
            # Entry: treat a NaN previous RSI as "previously above threshold" so that
            # an initially-oversold RSI can trigger an entry immediately.
            prev_check = (prev_rsi >= oversold) if not pd.isna(prev_rsi) else True
            if prev_check and (curr_rsi < oversold):
                pos = 1
        else:
            # pos == 1: Exit when RSI crosses above overbought
            prev_check_exit = (prev_rsi <= overbought) if not pd.isna(prev_rsi) else True
            if prev_check_exit and (curr_rsi > overbought):
                pos = 0

        positions.iloc[i] = pos

    # Ensure only 0/1 and no NaNs after warmup by filling any remaining NaNs with 0
    positions = positions.fillna(0).astype(int)

    return {"ohlcv": positions}
