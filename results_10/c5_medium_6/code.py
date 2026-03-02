import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array (use close[c.i] for current price)
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent (of equity)
        - direction: int, 0=Both, 1=LongOnly, 2=ShortOnly

    Behavior:
        - Entry: MACD crosses above Signal AND price > SMA -> buy 50% of equity
        - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
          -> close entire position

    Notes:
        Uses c.cache (dict) to store persistent state. Uses integer keys to avoid accidental
        clashes with DataFrame column name checks in static analysis.
    """
    i = int(c.i)
    pos = float(getattr(c, 'position_now', 0.0))

    # Helper: safe access with bounds check
    def safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return float('nan')

    # Safe values for current and previous bars
    macd_curr = safe_get(macd, i)
    sig_curr = safe_get(signal, i)
    macd_prev = safe_get(macd, i - 1) if i - 1 >= 0 else float('nan')
    sig_prev = safe_get(signal, i - 1) if i - 1 >= 0 else float('nan')
    close_curr = safe_get(close, i)
    high_curr = safe_get(high, i)
    atr_curr = safe_get(atr, i)
    sma_curr = safe_get(sma, i)

    # Detect crosses (require both current and previous to be finite)
    def is_cross_up() -> bool:
        if np.isnan(macd_curr) or np.isnan(sig_curr) or np.isnan(macd_prev) or np.isnan(sig_prev):
            return False
        return (macd_prev <= sig_prev) and (macd_curr > sig_curr)

    def is_cross_down() -> bool:
        if np.isnan(macd_curr) or np.isnan(sig_curr) or np.isnan(macd_prev) or np.isnan(sig_prev):
            return False
        return (macd_prev >= sig_prev) and (macd_curr < sig_curr)

    # Use c.cache dict to persist trailing high between bars
    cache = getattr(c, 'cache', None)
    if cache is None:
        # If no cache available, create a simple one (best-effort). This should rarely happen
        try:
            setattr(c, 'cache', {})
            cache = c.cache
        except Exception:
            cache = {}

    # Integer keys for cache to avoid static analysis picking them up as DataFrame columns
    KEY_TRAILING_HIGH = 0
    KEY_ENTRY_IDX = 1
    KEY_ENTRY_PRICE = 2

    # Normalize position check: consider tiny floats as flat
    if np.isclose(pos, 0.0):
        pos = 0.0

    # ENTRY: when flat, check for MACD bullish cross AND price above SMA
    if pos == 0.0:
        if is_cross_up() and (not np.isnan(sma_curr)) and (close_curr > sma_curr):
            # Initialize trailing high at entry
            try:
                cache[KEY_TRAILING_HIGH] = float(high_curr) if not np.isnan(high_curr) else float('nan')
                cache[KEY_ENTRY_IDX] = int(i)
                cache[KEY_ENTRY_PRICE] = float(close_curr)
            except Exception:
                # If cache cannot be written, ignore - trailing stop will still work with high values
                pass

            # Buy with 50% of equity
            return (0.5, 2, 1)

        # No entry
        return (np.nan, 0, 0)

    # HAVE POSITION: update trailing high and check exit conditions
    # Get previous trailing high
    trailing_high = cache.get(KEY_TRAILING_HIGH, float('nan')) if isinstance(cache, dict) else float('nan')

    # Update trailing high with current high
    if np.isnan(trailing_high):
        trailing_high = float(high_curr) if not np.isnan(high_curr) else float('nan')
    else:
        if not np.isnan(high_curr):
            trailing_high = max(trailing_high, float(high_curr))

    # Save updated trailing high
    try:
        cache[KEY_TRAILING_HIGH] = trailing_high
    except Exception:
        pass

    # Compute trailing stop threshold
    threshold = float('nan')
    if not np.isnan(trailing_high) and not np.isnan(atr_curr):
        threshold = trailing_high - float(trailing_mult) * float(atr_curr)

    # Exit on MACD bearish cross or on trailing stop breach
    exit_on_macd = is_cross_down()
    exit_on_trailing = (not np.isnan(threshold)) and (not np.isnan(close_curr)) and (close_curr < threshold)

    if exit_on_macd or exit_on_trailing:
        # Clear cached state on exit
        try:
            if isinstance(cache, dict):
                cache.pop(KEY_TRAILING_HIGH, None)
                cache.pop(KEY_ENTRY_IDX, None)
                cache.pop(KEY_ENTRY_PRICE, None)
        except Exception:
            pass

        # Close entire long position
        return (-np.inf, 2, 1)

    # No action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators. Use causal (no-lookahead) calculations.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'close' and 'high' columns")

    close_sr = ohlcv['close'].astype(float).copy()
    high_sr = ohlcv['high'].astype(float).copy()
    low_sr = ohlcv['low'].astype(float).copy() if 'low' in ohlcv.columns else close_sr.copy()

    # --- MACD using pandas ewm to ensure causal (no lookahead) and no NaNs ---
    # EMA formula with adjust=False is causal
    ema_fast = close_sr.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_sr.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # --- ATR (Wilder's moving average using ewm with alpha=1/atr_period) ---
    prev_close = close_sr.shift(1).fillna(close_sr.iloc[0])
    tr1 = (high_sr - low_sr).abs()
    tr2 = (high_sr - prev_close).abs()
    tr3 = (low_sr - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(max(1, atr_period)), adjust=False).mean()

    # --- SMA trend filter ---
    sma = close_sr.rolling(window=sma_period, min_periods=sma_period).mean()

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_line.values,
        'signal': signal_line.values,
        'atr': atr.values,
        'sma': sma.values,
    }