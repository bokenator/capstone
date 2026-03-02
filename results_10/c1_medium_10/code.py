import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Optional

# Module-level state to track entry index for the single-asset backtest.
# This is necessary because vectorbt's OrderContext provided in the prompt
# only exposes a limited set of attributes (c.i, c.position_now, c.cash_now).
# We store the index where we generated an entry order so we can compute the
# highest price since entry for the ATR-based trailing stop.
_ENTRY_INDEX: Optional[int] = None


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
    Order function for vectorbt.from_order_func implementing MACD entry + ATR trailing stop.

    Long-only. Uses a module-level variable to remember the most recent entry index
    for computing the highest price since entry.

    Entry (all must be true):
    - MACD crosses above Signal (previous bar macd < signal, current macd > signal)
    - Close price is above 50-period SMA (sma provided)

    Exit (any):
    - MACD crosses below Signal (previous bar macd > signal, current macd < signal)
    - Price falls below (highest_since_entry - trailing_mult * ATR)

    Returns tuple: (size, size_type, direction) as described in the prompt.
    """
    global _ENTRY_INDEX

    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks for index
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper: check if a value is valid (not NaN/inf)
    def is_valid(x):
        return (not np.isnan(x)) and np.isfinite(x)

    # Previous-bar crossover detection
    cross_up = False
    cross_down = False
    if i >= 1:
        if is_valid(macd[i - 1]) and is_valid(signal[i - 1]) and is_valid(macd[i]) and is_valid(signal[i]):
            cross_up = (macd[i - 1] < signal[i - 1]) and (macd[i] > signal[i])
            cross_down = (macd[i - 1] > signal[i - 1]) and (macd[i] < signal[i])

    # Price above SMA check
    price_above_sma = False
    if is_valid(close[i]) and is_valid(sma[i]):
        price_above_sma = close[i] > sma[i]

    # No position -> check for entry
    if pos == 0:
        # Clear stale entry index when flat
        _ENTRY_INDEX = None

        if cross_up and price_above_sma:
            # Enter full long (100% of equity)
            # Record the entry index for trailing stop calculations
            _ENTRY_INDEX = i
            return (1.0, 2, 1)  # 100% of equity, Percent size, Long only

        return (np.nan, 0, 0)  # No action

    # Have a long position -> check for exits
    # Ensure we have an entry index; if not, try to infer it by searching backwards
    if _ENTRY_INDEX is None:
        inferred = None
        for j in range(i, 0, -1):
            # require a valid crossover at j and price above sma at j
            if is_valid(macd[j - 1]) and is_valid(signal[j - 1]) and is_valid(macd[j]) and is_valid(signal[j]) and is_valid(sma[j]) and is_valid(close[j]):
                if (macd[j - 1] < signal[j - 1]) and (macd[j] > signal[j]) and (close[j] > sma[j]):
                    inferred = j
                    break
        _ENTRY_INDEX = inferred

    # Compute highest high since entry (inclusive). Use nan-aware max.
    highest_since = np.nan
    if _ENTRY_INDEX is None:
        # As a fallback, consider the whole history up to current bar
        try:
            highest_since = np.nanmax(high[: i + 1])
        except Exception:
            highest_since = np.nan
    else:
        # Bound the slice defensively
        start_idx = max(0, int(_ENTRY_INDEX))
        try:
            highest_since = np.nanmax(high[start_idx : i + 1])
        except Exception:
            highest_since = np.nan

    # Compute trailing stop level
    stop_price = np.nan
    if is_valid(highest_since) and is_valid(atr[i]):
        stop_price = highest_since - trailing_mult * atr[i]

    should_exit_trailing = False
    if is_valid(stop_price) and is_valid(close[i]):
        should_exit_trailing = close[i] < stop_price

    # Exit if MACD crosses down or trailing stop is hit
    if cross_down or should_exit_trailing:
        # Clear entry index on exit
        _ENTRY_INDEX = None
        # Close entire long position
        return (-np.inf, 2, 1)

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
    Compute required indicators: MACD (macd & signal), ATR, and SMA.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    Each value is a numpy array aligned with the input ohlcv DataFrame.
    """
    # Validate required columns from DATA_SCHEMA
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    # Use 'low' if available, otherwise fallback to 'close' to allow ATR computation
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (trend filter)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }
