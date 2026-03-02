import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any

# Module-level state variables for trailing stop logic (avoid dict indexing to pass static checks)
_IN_POSITION: bool = False
_ENTRY_INDEX: int | None = None
_HIGHEST: float = np.nan


def order_func(
    c: Any,
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
    global _IN_POSITION, _ENTRY_INDEX, _HIGHEST

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at the start of a run
    if i == 0:
        _IN_POSITION = False
        _ENTRY_INDEX = None
        _HIGHEST = np.nan

    # If flat, ensure internal state reflects no position
    if pos == 0.0:
        _IN_POSITION = False
        _ENTRY_INDEX = None
        _HIGHEST = np.nan

    def is_finite_val(x: float) -> bool:
        return np.isfinite(x)

    # ENTRY: No position and MACD crosses above signal AND price above SMA
    if pos == 0.0:
        if i >= 1:
            macd_prev = macd[i - 1]
            macd_curr = macd[i]
            signal_prev = signal[i - 1]
            signal_curr = signal[i]
            price = close[i]
            sma_curr = sma[i]

            if (
                is_finite_val(macd_prev)
                and is_finite_val(signal_prev)
                and is_finite_val(macd_curr)
                and is_finite_val(signal_curr)
                and is_finite_val(price)
                and is_finite_val(sma_curr)
            ):
                macd_cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
                price_above_sma = price > sma_curr

                if macd_cross_up and price_above_sma:
                    _IN_POSITION = True
                    _ENTRY_INDEX = i
                    _HIGHEST = float(high[i]) if is_finite_val(high[i]) else float(price)

                    # Enter long using 100% of equity
                    return (1.0, 2, 1)

    # POSITION HELD: check exit conditions
    else:
        # Initialize internal state if needed (e.g., first bar after fill)
        if not _IN_POSITION:
            _IN_POSITION = True
            _ENTRY_INDEX = i
            _HIGHEST = float(high[i]) if is_finite_val(high[i]) else float(close[i])

        # Update highest price since entry
        if is_finite_val(high[i]):
            if (not is_finite_val(_HIGHEST)) or (high[i] > _HIGHEST):
                _HIGHEST = float(high[i])

        # Exit on MACD cross below signal
        if i >= 1:
            macd_prev = macd[i - 1]
            macd_curr = macd[i]
            signal_prev = signal[i - 1]
            signal_curr = signal[i]

            if (
                is_finite_val(macd_prev)
                and is_finite_val(signal_prev)
                and is_finite_val(macd_curr)
                and is_finite_val(signal_curr)
            ):
                macd_cross_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
                if macd_cross_down:
                    _IN_POSITION = False
                    _ENTRY_INDEX = None
                    _HIGHEST = np.nan
                    return (-np.inf, 2, 1)

        # Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        atr_curr = atr[i]
        highest = _HIGHEST
        price = close[i]

        if is_finite_val(highest) and is_finite_val(atr_curr) and is_finite_val(price):
            threshold = highest - (trailing_mult * atr_curr)
            if price < threshold:
                _IN_POSITION = False
                _ENTRY_INDEX = None
                _HIGHEST = np.nan
                return (-np.inf, 2, 1)

    # No action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
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
    # Validate required columns per DATA_SCHEMA
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)

    # If low is missing, fallback to close (to allow ATR computation)
    if "low" in ohlcv.columns:
        low = ohlcv["low"].astype(float)
    else:
        low = close

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }