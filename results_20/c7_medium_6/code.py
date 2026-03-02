import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


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
    See prompt for detailed specification.
    """
    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position (0.0 if flat)

    # Initialize per-backtest state on the function object. This allows us to
    # track the highest price since entry for the trailing stop without using
    # global variables. The state is reset at the start of each run (i == 0).
    if not hasattr(order_func, "_state"):
        # Use a list to avoid accidental static analysis that treats string keys
        # as DataFrame column access. state[0] will hold the highest price since entry.
        order_func._state = [None]

    state = order_func._state

    # Reset state at the beginning of a backtest/run
    if i == 0:
        order_func._state = [None]
        state = order_func._state

    def _is_valid(arr: np.ndarray, idx: int) -> bool:
        """Return True if arr[idx] is a finite number."""
        try:
            v = arr[idx]
        except Exception:
            return False
        return not (np.isnan(v) or np.isinf(v))

    # Entry logic (long-only)
    if pos == 0.0:
        should_enter = False

        # MACD bullish crossover: macd crosses above signal at current bar
        if i > 0:
            if (
                _is_valid(macd, i - 1)
                and _is_valid(signal, i - 1)
                and _is_valid(macd, i)
                and _is_valid(signal, i)
            ):
                macd_prev = float(macd[i - 1])
                sig_prev = float(signal[i - 1])
                macd_cur = float(macd[i])
                sig_cur = float(signal[i])

                macd_cross_up = (macd_prev <= sig_prev) and (macd_cur > sig_cur)

                # Trend filter: price must be above SMA
                price_above_sma = (not np.isnan(sma[i])) and (float(close[i]) > float(sma[i]))

                if macd_cross_up and price_above_sma:
                    should_enter = True

        if should_enter:
            # Initialize highest price since entry to the current high (or close if high is NaN)
            current_high = float(high[i]) if (not np.isnan(high[i])) else float(close[i])
            state[0] = current_high

            # Use 50% of equity to enter (percent size)
            return (0.5, 2, 1)

    else:
        # We're in a position: update highest_since_entry using current high
        if state[0] is None:
            # If for some reason highest wasn't set (e.g. resumed mid-position), initialize it
            state[0] = float(high[i]) if (not np.isnan(high[i])) else float(close[i])
        else:
            if not np.isnan(high[i]):
                state[0] = float(max(state[0], float(high[i])))

        # MACD bearish crossover: macd crosses below signal at current bar
        macd_cross_down = False
        if i > 0 and (
            _is_valid(macd, i - 1)
            and _is_valid(signal, i - 1)
            and _is_valid(macd, i)
            and _is_valid(signal, i)
        ):
            macd_cross_down = (float(macd[i - 1]) >= float(signal[i - 1])) and (float(macd[i]) < float(signal[i]))

        # ATR-based trailing stop
        trailing_stop_trigger = False
        if state[0] is not None and _is_valid(atr, i):
            threshold = state[0] - float(trailing_mult) * float(atr[i])
            if not np.isnan(close[i]) and float(close[i]) < threshold:
                trailing_stop_trigger = True

        if macd_cross_down or trailing_stop_trigger:
            # Reset highest for the next position
            state[0] = None
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
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are 1-d numpy arrays of the same length as the input ohlcv.
    """
    # Validate required columns (per DATA_SCHEMA: 'high' and 'close' are required)
    if 'close' not in ohlcv.columns:
        raise KeyError("DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("DataFrame must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # If 'low' is missing, fallback to using 'close' so ATR can still be computed
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low']
    else:
        low_series = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values.astype(float),
        'high': high_series.values.astype(float),
        'macd': macd_ind.macd.values.astype(float),
        'signal': macd_ind.signal.values.astype(float),
        'atr': atr_ind.atr.values.astype(float),
        'sma': sma_ind.ma.values.astype(float),
    }