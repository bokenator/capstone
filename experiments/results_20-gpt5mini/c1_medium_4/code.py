import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Optional

# Module-level state to track entry index and highest price since entry for the single-asset strategy.
# These are used only by order_func and persist across calls (required because OrderContext
# doesn't expose the entry index). This keeps the implementation simple for a single-asset
# backtest as required by the prompt.
_ENTRY_INDEX: Optional[int] = None
_HIGHEST_PRICE: Optional[float] = None
_PREV_POS: float = 0.0


def order_func(
    c: Any,
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
    """
    global _ENTRY_INDEX, _HIGHEST_PRICE, _PREV_POS

    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Helper functions for cross detection with NaN-safety
    def cross_up(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx - 1], arr_b[idx - 1]
        a1, b1 = arr_a[idx], arr_b[idx]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 <= b0) and (a1 > b1)

    def cross_down(arr_a: np.ndarray, arr_b: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = arr_a[idx - 1], arr_b[idx - 1]
        a1, b1 = arr_a[idx], arr_b[idx]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 >= b0) and (a1 < b1)

    # Detect transitions to maintain entry index and highest price since entry
    # If we just opened a position (pos > 0 and previously flat), mark entry
    if pos > 0 and _PREV_POS == 0:
        _ENTRY_INDEX = i
        # Initialize highest price using available high (fallback to close if NaN)
        current_high = float(high[i]) if not np.isnan(high[i]) else float(close[i])
        _HIGHEST_PRICE = current_high

    # If position was closed since last bar, clear state
    if pos == 0 and _PREV_POS > 0:
        _ENTRY_INDEX = None
        _HIGHEST_PRICE = None

    # ENTRY: only when flat
    if pos == 0:
        # Entry condition: MACD crosses above Signal AND price above SMA
        should_enter = False
        if cross_up(macd, signal, i):
            # Ensure SMA exists and is not NaN
            if not np.isnan(sma[i]) and not np.isnan(close[i]):
                if close[i] > sma[i]:
                    should_enter = True
        if should_enter:
            # Buy with 50% of equity (Percent sizing)
            # size_type=2 -> Percent, direction=1 -> LongOnly
            _PREV_POS = pos
            return (0.5, 2, 1)

    # MANAGE OPEN POSITION
    if pos > 0:
        # Update highest price since entry
        if _HIGHEST_PRICE is None:
            _HIGHEST_PRICE = float(high[i]) if not np.isnan(high[i]) else float(close[i])
        else:
            # Use high price to track the highest value since entry
            if not np.isnan(high[i]):
                _HIGHEST_PRICE = max(_HIGHEST_PRICE, float(high[i]))

        # Exit conditions
        should_exit = False

        # 1) MACD crosses below Signal
        if cross_down(macd, signal, i):
            should_exit = True

        # 2) Price falls below (highest_since_entry - trailing_mult * ATR)
        # Require ATR and highest price to be available
        if _HIGHEST_PRICE is not None and not np.isnan(atr[i]):
            trail_price = _HIGHEST_PRICE - (trailing_mult * float(atr[i]))
            if not np.isnan(close[i]) and close[i] < trail_price:
                should_exit = True

        if should_exit:
            # Close entire long position
            # Using (-np.inf, 2, 1) signals to close the entire long position
            _ENTRY_INDEX = None
            _HIGHEST_PRICE = None
            _PREV_POS = pos
            return (-np.inf, 2, 1)

    # Update previous position for next call
    _PREV_POS = pos

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
    Precompute all indicators. Use vectorbt indicator classes.

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
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'high' column")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']

    # Low is optional in DATA_SCHEMA; if missing, fallback to close (best-effort)
    if 'low' in ohlcv.columns:
        low_sr = ohlcv['low']
    else:
        low_sr = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # Compute SMA (moving average)
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    return {
        'close': close_sr.values.astype(float),
        'high': high_sr.values.astype(float),
        'macd': macd_ind.macd.values.astype(float),
        'signal': macd_ind.signal.values.astype(float),
        'atr': atr_ind.atr.values.astype(float),
        'sma': sma_ind.ma.values.astype(float),
    }