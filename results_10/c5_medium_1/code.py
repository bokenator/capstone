import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Optional

# Module-level state to track entry/highest price across bars within a single backtest run.
# Use simple module-level variables (avoid dict-style string keys to satisfy static checks).
_ORDER_ENTRY_INDEX: Optional[int] = None
_ORDER_HIGHEST: float = np.nan


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
    """
    global _ORDER_ENTRY_INDEX, _ORDER_HIGHEST

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the start of each backtest run
    if i == 0:
        _ORDER_ENTRY_INDEX = None
        _ORDER_HIGHEST = np.nan

    # Safety bounds checks
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # If flat, clear any stored entry state
    if pos == 0.0:
        _ORDER_ENTRY_INDEX = None
        _ORDER_HIGHEST = np.nan

        # ENTRY CONDITIONS (all must be true):
        # - MACD line crosses above Signal line
        # - Price is above 50-period SMA
        should_enter = False

        # Detect MACD bullish crossover at current bar (no lookahead)
        if i > 0:
            prev_macd = macd[i - 1]
            prev_sig = signal[i - 1]
            cur_macd = macd[i]
            cur_sig = signal[i]

            if (
                not np.isnan(prev_macd)
                and not np.isnan(prev_sig)
                and not np.isnan(cur_macd)
                and not np.isnan(cur_sig)
                and cur_macd > cur_sig
                and prev_macd <= prev_sig
            ):
                # Trend filter: price above SMA
                if not np.isnan(close[i]) and not np.isnan(sma[i]) and close[i] > sma[i]:
                    should_enter = True

        if should_enter:
            # Record entry index and initialize highest price since entry with current high
            _ORDER_ENTRY_INDEX = i
            _ORDER_HIGHEST = high[i] if not np.isnan(high[i]) else close[i]

            # Buy with 50% of equity (long-only)
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    else:
        # We have a position - update highest_since_entry and check exits
        # If we previously recorded entry index => use it; otherwise attempt to initialize
        if _ORDER_ENTRY_INDEX is None:
            # Fallback: initialize entry index to the current bar when state was missed
            _ORDER_ENTRY_INDEX = i
            _ORDER_HIGHEST = high[i] if not np.isnan(high[i]) else close[i]

        # Update highest price since entry
        cur_high = high[i] if not np.isnan(high[i]) else close[i]
        if np.isnan(_ORDER_HIGHEST) or cur_high > _ORDER_HIGHEST:
            _ORDER_HIGHEST = cur_high

        # Compute trailing stop: highest_since_entry - trailing_mult * ATR
        trailing_stop_hit = False
        if not np.isnan(_ORDER_HIGHEST) and not np.isnan(atr[i]):
            trailing_stop_price = _ORDER_HIGHEST - float(trailing_mult) * float(atr[i])
            if not np.isnan(close[i]) and close[i] < trailing_stop_price:
                trailing_stop_hit = True

        # MACD bearish cross exit
        macd_cross_down = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_sig = signal[i - 1]
            cur_macd = macd[i]
            cur_sig = signal[i]

            if (
                not np.isnan(prev_macd)
                and not np.isnan(prev_sig)
                and not np.isnan(cur_macd)
                and not np.isnan(cur_sig)
                and cur_macd < cur_sig
                and prev_macd >= prev_sig
            ):
                macd_cross_down = True

        if macd_cross_down or trailing_stop_hit:
            # Signal to close entire long position
            # Clear stored state; it will also be cleared at next bar when pos==0
            _ORDER_ENTRY_INDEX = None
            _ORDER_HIGHEST = np.nan

            # Close entire long position by using -inf percent (vectorbt convention)
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
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
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'high' and 'close' columns")

    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)

    # Use 'low' if available, otherwise fall back to 'close' to avoid errors
    if "low" in ohlcv.columns:
        low_sr = ohlcv["low"].astype(float)
    else:
        low_sr = close_sr

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    return {
        "close": close_sr.values,
        "high": high_sr.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }