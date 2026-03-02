import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Module-level state to track highest price since entry for the trailing stop.
# This will be reinitialized at the start of each backtest run (when c.i == 0).
_ENTRY_HIGH: float = -np.inf


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
    Order generation function for vectorbt.from_order_func (no numba).

    Implements:
      - Entry: MACD crosses above signal AND close > SMA
      - Exit: MACD crosses below signal OR close < (highest_since_entry - trailing_mult * ATR)

    Long-only. Uses 50% of equity for entries. Closes full position on exit.

    Args and returns documented in the prompt.
    """
    global _ENTRY_HIGH

    i = int(c.i)
    pos = float(c.position_now)

    # Reinitialize state at the start of a run
    if i == 0:
        _ENTRY_HIGH = -np.inf

    # Safety: ensure indices are within bounds
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper to check valid numeric value
    def is_valid(x):
        return not (x is None or (isinstance(x, float) and np.isnan(x)))

    # No position: check for entry
    if pos == 0.0:
        # Need previous bar to detect crossover
        if i > 0:
            if (
                is_valid(macd[i]) and is_valid(signal[i]) and
                is_valid(macd[i - 1]) and is_valid(signal[i - 1])
            ):
                macd_cross_up = (macd[i - 1] <= signal[i - 1]) and (macd[i] > signal[i])
            else:
                macd_cross_up = False

            price_above_sma = is_valid(sma[i]) and is_valid(close[i]) and (close[i] > sma[i])

            if macd_cross_up and price_above_sma:
                # Set initial highest price since entry to current high (fallback to close)
                _ENTRY_HIGH = high[i] if is_valid(high[i]) else close[i]
                # Enter with 50% of equity
                return (0.5, 2, 1)

    else:
        # We have a position: update highest_since_entry
        if is_valid(high[i]):
            if _ENTRY_HIGH == -np.inf:
                _ENTRY_HIGH = high[i]
            else:
                # Use the high of the bar to track new highs
                try:
                    _ENTRY_HIGH = max(_ENTRY_HIGH, float(high[i]))
                except Exception:
                    # Fallback if types are unexpected
                    _ENTRY_HIGH = max(_ENTRY_HIGH, high[i])

        # Check for MACD bearish cross (exit)
        macd_cross_down = False
        if i > 0:
            if (
                is_valid(macd[i]) and is_valid(signal[i]) and
                is_valid(macd[i - 1]) and is_valid(signal[i - 1])
            ):
                macd_cross_down = (macd[i - 1] >= signal[i - 1]) and (macd[i] < signal[i])

        if macd_cross_down:
            # Reset trailing state and close position
            _ENTRY_HIGH = -np.inf
            return (-np.inf, 2, 1)

        # Trailing stop check: require ATR and a valid entry high
        if is_valid(atr[i]) and (_ENTRY_HIGH != -np.inf) and is_valid(close[i]):
            stop_price = _ENTRY_HIGH - float(trailing_mult) * float(atr[i])
            # Exit if price falls strictly below the trailing stop
            if close[i] < stop_price:
                _ENTRY_HIGH = -np.inf
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
    Compute required indicators using vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain a 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain a 'high' column")

    close_ser = ohlcv['close'].astype(float)
    high_ser = ohlcv['high'].astype(float)

    # low is optional in DATA_SCHEMA; fallback to close if missing
    if 'low' in ohlcv.columns:
        low_ser = ohlcv['low'].astype(float)
    else:
        # Use close as conservative fallback for ATR calculation if low not provided
        low_ser = close_ser

    # MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        'close': close_ser.values,
        'high': high_ser.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }