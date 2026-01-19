"""
Reference Implementation: Medium Strategy (MACD + ATR Trailing Stop)
=====================================================================

Strategy:
- MACD crossover for entry signals
- 50-period SMA as trend filter
- ATR-based trailing stop for exits

Entry conditions (all must be true):
- MACD line crosses above Signal line
- Price is above 50-period SMA

Exit conditions (any):
- MACD line crosses below Signal line
- Price falls below (highest_since_entry - trailing_mult * ATR)

Function signatures:
    compute_indicators(ohlcv, **params) -> dict[str, np.ndarray]
    order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


# Module-level state for tracking position across bars
_state = {
    "in_position": False,
    "highest_since_entry": -np.inf,
}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Compute all indicators required by the strategy.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR calculation period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.
    """
    global _state

    # Reset state for fresh backtest
    _state = {
        "in_position": False,
        "highest_since_entry": -np.inf,
    }

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # Compute MACD using vectorbt
    macd_ind = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal
    )

    # Compute SMA using vectorbt
    sma = vbt.MA.run(close, window=sma_period).ma

    # Compute ATR using vectorbt
    atr = vbt.ATR.run(high, low, close, window=atr_period).atr

    return {
        "close": close.values.astype(float),
        "high": high.values.astype(float),
        "macd": macd_ind.macd.values.astype(float),
        "signal": macd_ind.signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
    }


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    Args:
        c: vectorbt OrderContext with:
           - c.i: current bar index (int)
           - c.position_now: current position size (float)
           - c.cash_now: current cash balance (float)
        close: Close prices array
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        Tuple of (size, size_type, direction):
        - size: float (np.nan = no action)
        - size_type: int (0=Amount, 1=Value, 2=Percent)
        - direction: int (0=Both, 1=LongOnly, 2=ShortOnly)
    """
    global _state

    i = c.i
    pos = c.position_now

    # Current values
    price = close[i]
    high_price = high[i]
    macd_val = macd[i]
    signal_val = signal[i]
    atr_val = atr[i]
    sma_val = sma[i]

    # Previous values for crossover detection
    if i > 0:
        macd_prev = macd[i - 1]
        signal_prev = signal[i - 1]
    else:
        macd_prev = np.nan
        signal_prev = np.nan

    # Skip if indicators not ready
    if np.isnan(macd_val) or np.isnan(signal_val) or np.isnan(sma_val):
        return (np.nan, 0, 0)

    # Sync state with actual position
    if pos > 0 and not _state["in_position"]:
        _state["in_position"] = True
        _state["highest_since_entry"] = high_price
    elif pos == 0 and _state["in_position"]:
        _state["in_position"] = False
        _state["highest_since_entry"] = -np.inf

    # Update highest price since entry
    if _state["in_position"]:
        if high_price > _state["highest_since_entry"]:
            _state["highest_since_entry"] = high_price

    # Entry logic: no position
    if pos == 0:
        # Check for MACD crossover (MACD crosses above signal)
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            macd_cross_up = (macd_prev <= signal_prev) and (macd_val > signal_val)

            # Entry: MACD cross up AND price above SMA
            if macd_cross_up and price > sma_val:
                _state["in_position"] = True
                _state["highest_since_entry"] = high_price
                # Buy with 95% of equity (long only)
                return (0.95, 2, 1)

    # Exit logic: have position
    else:
        should_exit = False

        # Exit condition 1: MACD crosses below signal
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            macd_cross_down = (macd_prev >= signal_prev) and (macd_val < signal_val)
            if macd_cross_down:
                should_exit = True

        # Exit condition 2: Trailing stop hit
        if not np.isnan(atr_val) and _state["highest_since_entry"] > -np.inf:
            stop_price = _state["highest_since_entry"] - (trailing_mult * atr_val)
            if price < stop_price:
                should_exit = True

        if should_exit:
            _state["in_position"] = False
            _state["highest_since_entry"] = -np.inf
            # Close entire position
            return (-np.inf, 2, 1)

    # No action
    return (np.nan, 0, 0)
