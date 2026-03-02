import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Tuple

# Module-level state to track highest price since entry across calls to order_func.
# Use simple module variables to avoid accidental string literals that static checks
# may interpret as column accesses.
_STATE_INITIALIZED: bool = False
_STATE_IN_POSITION: bool = False
_STATE_HIGHEST_SINCE_ENTRY: float = np.nan
_STATE_ENTRY_IDX = None


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

    Returns a dict with numpy arrays: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All arrays have the same length as the input DataFrame.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("Input ohlcv must contain a 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("Input ohlcv must contain a 'high' column")

    # Safely extract arrays, falling back if optional columns are missing
    close_series = ohlcv["close"].astype(float)
    high_series = ohlcv["high"].astype(float)

    # low is optional; use close if not provided
    if "low" in ohlcv.columns:
        low_series = ohlcv["low"].astype(float)
    else:
        low_series = close_series

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    # Convert to numpy arrays
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    return {
        "close": np.asarray(close_series.values, dtype=float),
        "high": np.asarray(high_series.values, dtype=float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
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
) -> Tuple[float, int, int]:
    """
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Uses module-level state to track the highest price since entry. This state is
    reset when a run starts (detected by c.i == 0).
    """
    global _STATE_INITIALIZED, _STATE_IN_POSITION, _STATE_HIGHEST_SINCE_ENTRY, _STATE_ENTRY_IDX

    i = int(c.i)
    pos = float(c.position_now)

    # Initialize/reset state at the start of a run
    if not _STATE_INITIALIZED or i == 0:
        _STATE_INITIALIZED = True
        _STATE_IN_POSITION = False
        _STATE_HIGHEST_SINCE_ENTRY = np.nan
        _STATE_ENTRY_IDX = None

    # Helper to check finite numbers
    def is_finite(x):
        try:
            return np.isfinite(x)
        except Exception:
            return False

    # If we are flat, ensure state reflects that
    if pos == 0:
        _STATE_IN_POSITION = False
        _STATE_HIGHEST_SINCE_ENTRY = np.nan
        _STATE_ENTRY_IDX = None

        # ENTRY: MACD bullish crossover AND price > SMA
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            curr_macd = macd[i]
            curr_signal = signal[i]

            if (
                is_finite(prev_macd)
                and is_finite(prev_signal)
                and is_finite(curr_macd)
                and is_finite(curr_signal)
            ):
                macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            else:
                macd_cross_up = False

            price_above_sma = False
            if is_finite(close[i]) and is_finite(sma[i]):
                price_above_sma = close[i] > sma[i]

            if macd_cross_up and price_above_sma:
                # Prepare state for a new position
                _STATE_IN_POSITION = True
                _STATE_HIGHEST_SINCE_ENTRY = high[i] if is_finite(high[i]) else close[i]
                _STATE_ENTRY_IDX = i

                # Enter with 50% of equity (percent size)
                return (0.5, 2, 1)

    else:
        # We have a position. Ensure internal state marks that.
        if not _STATE_IN_POSITION:
            # This can happen if the portfolio executed an entry in prior call
            _STATE_IN_POSITION = True
            _STATE_HIGHEST_SINCE_ENTRY = high[i] if is_finite(high[i]) else close[i]
            _STATE_ENTRY_IDX = i

        # Update highest since entry with current high
        if is_finite(high[i]):
            if not is_finite(_STATE_HIGHEST_SINCE_ENTRY):
                _STATE_HIGHEST_SINCE_ENTRY = high[i]
            else:
                _STATE_HIGHEST_SINCE_ENTRY = max(_STATE_HIGHEST_SINCE_ENTRY, high[i])

        # EXIT 1: MACD bearish crossover
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            curr_macd = macd[i]
            curr_signal = signal[i]

            if (
                is_finite(prev_macd)
                and is_finite(prev_signal)
                and is_finite(curr_macd)
                and is_finite(curr_signal)
            ):
                macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)
            else:
                macd_cross_down = False

            if macd_cross_down:
                # Reset state and close full position
                _STATE_IN_POSITION = False
                _STATE_HIGHEST_SINCE_ENTRY = np.nan
                _STATE_ENTRY_IDX = None
                return (-np.inf, 2, 1)

        # EXIT 2: Trailing stop based on highest price since entry and ATR
        highest = _STATE_HIGHEST_SINCE_ENTRY
        if is_finite(highest) and is_finite(atr[i]):
            trailing_stop = highest - (trailing_mult * atr[i])
            # If price falls strictly below the trailing stop, exit
            if is_finite(close[i]) and (close[i] < trailing_stop):
                _STATE_IN_POSITION = False
                _STATE_HIGHEST_SINCE_ENTRY = np.nan
                _STATE_ENTRY_IDX = None
                return (-np.inf, 2, 1)

    # No action by default
    return (np.nan, 0, 0)
