import numpy as np
import pandas as pd
import vectorbt as vbt

# Global state to track entry index and highest price since entry
_ENTRY_IDX = None  # type: ignore
_HIGHEST = np.nan


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
    global _ENTRY_IDX, _HIGHEST

    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks for index bounds
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Reset state at the start of the run
    if i == 0:
        _ENTRY_IDX = None
        _HIGHEST = np.nan

    # Current values
    close_now = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_now = float(high[i]) if not np.isnan(high[i]) else np.nan
    macd_now = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_now = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    atr_now = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_now = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # Previous values for crossover detection
    macd_prev = float(macd[i - 1]) if i > 0 and not np.isnan(macd[i - 1]) else np.nan
    signal_prev = float(signal[i - 1]) if i > 0 and not np.isnan(signal[i - 1]) else np.nan

    macd_cross_up = False
    macd_cross_down = False

    if not np.isnan(macd_prev) and not np.isnan(signal_prev) and not np.isnan(macd_now) and not np.isnan(signal_now):
        macd_cross_up = (macd_prev <= signal_prev) and (macd_now > signal_now)
        macd_cross_down = (macd_prev >= signal_prev) and (macd_now < signal_now)

    # ENTRY: No position and MACD crosses above signal and price above SMA
    if pos == 0.0:
        if macd_cross_up and (not np.isnan(sma_now)) and (not np.isnan(close_now)) and (close_now > sma_now):
            # Record entry index and initialize highest price since entry
            _ENTRY_IDX = i
            _HIGHEST = high_now if not np.isnan(high_now) else close_now
            # Use 50% of equity to enter (percent size)
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # UPDATE highest_since_entry while in position
    if pos != 0.0:
        if _ENTRY_IDX is None:
            # If entry index wasn't recorded for some reason, initialize it now
            _ENTRY_IDX = i
            _HIGHEST = high_now if not np.isnan(high_now) else close_now
        else:
            # Update running highest price since entry
            if np.isnan(_HIGHEST):
                _HIGHEST = high_now if not np.isnan(high_now) else _HIGHEST
            else:
                if not np.isnan(high_now) and high_now > _HIGHEST:
                    _HIGHEST = high_now

        highest_since_entry = _HIGHEST

        # Compute trailing stop level if ATR and highest are available
        trailing_level = np.nan
        if (not np.isnan(atr_now)) and (not np.isnan(highest_since_entry)):
            trailing_level = highest_since_entry - trailing_mult * atr_now

        # EXIT: MACD crosses below OR close price falls below trailing stop level
        if macd_cross_down:
            # Clear state and close entire position
            _ENTRY_IDX = None
            _HIGHEST = np.nan
            # Close by specifying negative current position amount (shares)
            return (-pos, 0, 1)

        if (not np.isnan(trailing_level)) and (not np.isnan(close_now)) and (close_now < trailing_level):
            _ENTRY_IDX = None
            _HIGHEST = np.nan
            return (-pos, 0, 1)

    # No action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
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
    # Validate required columns
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"compute_indicators: required column '{col}' not found in ohlcv")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]
    low_series = ohlcv["low"]

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (simple moving average)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }