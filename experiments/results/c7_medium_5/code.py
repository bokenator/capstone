import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state for tracking the current open position across sequential calls
# This state is reset at the first bar of each backtest run (i == 0)
_in_position = False
_entry_high = np.nan
_entry_idx = -1


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

    Return Examples:
        (100.0, 0, 1)     # Buy 100 shares, long only
        (0.5, 2, 1)       # Buy with 50% of equity, long only
        (-np.inf, 2, 1)   # Close entire long position (size=-inf with Percent)
        (np.nan, 0, 0)    # No action (size=nan means no order)
    """
    global _in_position, _entry_high, _entry_idx

    i = int(c.i)
    pos = float(c.position_now)

    # Reset module-level state at the start of a run
    if i == 0:
        _in_position = False
        _entry_high = np.nan
        _entry_idx = -1

    # Safety guards: validate index range
    if i < 0 or i >= len(close):
        return vbt.portfolio.enums.NoOrder

    price = float(close[i])
    hi = float(high[i])

    # Previous values for crossover detection
    if i == 0:
        macd_prev = np.nan
        signal_prev = np.nan
    else:
        macd_prev = float(macd[i - 1])
        signal_prev = float(signal[i - 1])

    macd_i = float(macd[i])
    signal_i = float(signal[i])
    atr_i = float(atr[i])
    sma_i = float(sma[i])

    # Determine MACD cross directions (use previous bar to avoid lookahead)
    macd_cross_up = False
    macd_cross_down = False
    if np.isfinite(macd_prev) and np.isfinite(signal_prev) and np.isfinite(macd_i) and np.isfinite(signal_i):
        macd_cross_up = (macd_prev <= signal_prev) and (macd_i > signal_i)
        macd_cross_down = (macd_prev >= signal_prev) and (macd_i < signal_i)

    # ENTRY: Only when flat and MACD crosses up and price is above SMA
    if pos == 0.0:
        if macd_cross_up and np.isfinite(sma_i) and np.isfinite(atr_i) and np.isfinite(price):
            # Trend filter: price must be above 50-period SMA
            if price > sma_i:
                # Enter long with 100% of equity
                _in_position = True
                _entry_idx = i
                _entry_high = hi if np.isfinite(hi) else price
                try:
                    return vbt.portfolio.nb.order_nb(1.0, 2, 1)
                except Exception:
                    # Fallback: return tuple as legacy format
                    return (1.0, 2, 1)

        return vbt.portfolio.enums.NoOrder

    # IN POSITION: manage trailing stop and MACD bearish cross
    else:
        # If we detect a position but internal state wasn't set (edge cases), initialize
        if not _in_position:
            _in_position = True
            _entry_idx = i
            _entry_high = hi if np.isfinite(hi) else price

        # Update highest price since entry using the current bar's high
        if np.isfinite(hi) and (not np.isfinite(_entry_high) or hi > _entry_high):
            _entry_high = hi

        # Calculate trailing stop level
        stop_price = np.nan
        if np.isfinite(_entry_high) and np.isfinite(atr_i):
            stop_price = _entry_high - (trailing_mult * atr_i)

        # EXIT on MACD bearish cross
        if macd_cross_down:
            # Clear state and close position
            _in_position = False
            _entry_idx = -1
            _entry_high = np.nan
            try:
                return vbt.portfolio.nb.order_nb(-np.inf, 2, 1)
            except Exception:
                return (-np.inf, 2, 1)

        # EXIT on ATR-based trailing stop breach (price falls below stop)
        if np.isfinite(stop_price) and np.isfinite(price) and (price < stop_price):
            _in_position = False
            _entry_idx = -1
            _entry_high = np.nan
            try:
                return vbt.portfolio.nb.order_nb(-np.inf, 2, 1)
            except Exception:
                return (-np.inf, 2, 1)

        return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict:
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
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # If low column is missing, fallback to close (best-effort). Tests typically provide low.
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else close_series

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }