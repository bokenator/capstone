import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


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
    """

    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position (0.0 if flat)

    # Initialize or reset persistent state on first bar of a run
    # Use function attributes to track state across calls within a single backtest run
    if not hasattr(order_func, "_is_initialized") or i == 0:
        order_func._is_initialized = True
        order_func.entry_index = None
        order_func.highest_since_entry = np.nan

    # If flat - clear any entry state
    if pos == 0.0:
        order_func.entry_index = None
        order_func.highest_since_entry = np.nan

        # Entry logic: need a MACD bullish cross (prev <=, curr >) and price above SMA
        # Require at least one prior bar to detect crossover
        if i == 0:
            return (np.nan, 0, 0)

        macd_prev = macd[i - 1]
        macd_curr = macd[i]
        sig_prev = signal[i - 1]
        sig_curr = signal[i]
        price = close[i]
        sma_val = sma[i]

        # Guard against NaNs in indicators
        if (
            np.isnan(macd_prev)
            or np.isnan(macd_curr)
            or np.isnan(sig_prev)
            or np.isnan(sig_curr)
            or np.isnan(sma_val)
            or np.isnan(price)
        ):
            return (np.nan, 0, 0)

        bullish_cross = (macd_prev <= sig_prev) and (macd_curr > sig_curr)
        in_uptrend = price > sma_val

        if bullish_cross and in_uptrend:
            # Record entry and initialize highest price since entry
            order_func.entry_index = i
            order_func.highest_since_entry = high[i] if not np.isnan(high[i]) else price

            # Enter using 100% of equity (long-only)
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    else:
        # We have a position - update highest_since_entry
        curr_high = high[i] if not np.isnan(high[i]) else close[i]
        if np.isnan(order_func.highest_since_entry):
            order_func.highest_since_entry = curr_high
        else:
            if not np.isnan(curr_high) and curr_high > order_func.highest_since_entry:
                order_func.highest_since_entry = curr_high

        price = close[i]

        # 1) Exit on MACD bearish cross
        if i > 0:
            macd_prev = macd[i - 1]
            macd_curr = macd[i]
            sig_prev = signal[i - 1]
            sig_curr = signal[i]

            if not (
                np.isnan(macd_prev)
                or np.isnan(macd_curr)
                or np.isnan(sig_prev)
                or np.isnan(sig_curr)
            ):
                bearish_cross = (macd_prev >= sig_prev) and (macd_curr < sig_curr)
                if bearish_cross:
                    # Close entire long position
                    return (-np.inf, 2, 1)

        # 2) Exit on ATR-based trailing stop: price < (highest_since_entry - trailing_mult * ATR)
        if not np.isnan(order_func.highest_since_entry) and not np.isnan(atr[i]):
            trailing_stop = order_func.highest_since_entry - trailing_mult * atr[i]
            if price < trailing_stop:
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

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are np.ndarray of same length as input.
    """

    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' and 'close' columns")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    # Use 'low' if available for ATR, otherwise fall back to 'close' to avoid errors
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # MACD
    macd_ind = vbt.MACD.run(
        close_series,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    # Extract numpy arrays and ensure consistent dtype
    close_arr = np.asarray(close_series.values, dtype=float)
    high_arr = np.asarray(high_series.values, dtype=float)
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    # Final dict
    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }