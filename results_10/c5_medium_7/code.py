import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple
import types


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt.from_order_func (NO NUMBA).

    Implements MACD crossover entries with ATR-based trailing stops.

    Entry (when flat):
      - MACD crosses above Signal at current bar
      - Price (close) is above 50-period SMA
      -> Enter long with 100% of equity (size_type=2, size=1.0)

    Exit (when in position):
      - MACD crosses below Signal OR
      - Price falls below (highest_since_entry - trailing_mult * ATR)
      -> Close entire long position (size=-np.inf, size_type=2)

    State is kept on the function object to track highest price since entry.
    The state is reset at the start of each run (when c.i == 0).

    Args:
        c: vectorbt OrderContext (has attributes c.i, c.position_now, c.cash_now)
        close, high, macd, signal, atr, sma: numpy arrays of indicators
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        (size, size_type, direction) tuple as specified in the prompt.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize / reset state on first bar of a run
    state = getattr(order_func, "_state", None)
    if state is None or i == 0:
        # Use a SimpleNamespace to avoid bracket string indexing that static checks may
        # confuse with dataframe column access.
        state = types.SimpleNamespace(
            in_position=False,  # Whether we believe we're in a long position
            entry_i=None,       # Index of entry
            highest=-np.inf,    # Highest high since entry
            entry_price=np.nan, # Entry price
        )
        order_func._state = state

    # Helper to detect MACD crosses without lookahead
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a1, b1 = macd[idx - 1], signal[idx - 1]
        a2, b2 = macd[idx], signal[idx]
        if np.isnan(a1) or np.isnan(b1) or np.isnan(a2) or np.isnan(b2):
            return False
        return (a1 <= b1) and (a2 > b2)

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a1, b1 = macd[idx - 1], signal[idx - 1]
        a2, b2 = macd[idx], signal[idx]
        if np.isnan(a1) or np.isnan(b1) or np.isnan(a2) or np.isnan(b2):
            return False
        return (a1 >= b1) and (a2 < b2)

    # Ensure arrays have valid values at i
    close_i = close[i] if i < len(close) else np.nan
    high_i = high[i] if i < len(high) else np.nan
    atr_i = atr[i] if i < len(atr) else np.nan
    sma_i = sma[i] if i < len(sma) else np.nan

    # If flat, consider entry
    if abs(pos) < 1e-12:
        # Make sure indicators are present
        if macd_cross_up(i) and (not np.isnan(sma_i)) and (not np.isnan(close_i)):
            if close_i > sma_i:
                # Prepare state for new position
                state.in_position = True
                state.entry_i = i
                state.entry_price = close_i
                state.highest = high_i if (not np.isnan(high_i)) else close_i

                # Enter long with 100% of equity
                return (1.0, 2, 1)

    else:
        # We have a position according to the portfolio
        # If state was lost for any reason, initialize from current bar
        if not getattr(state, "in_position", False):
            state.in_position = True
            state.entry_i = i
            state.entry_price = close_i
            state.highest = high_i if (not np.isnan(high_i)) else close_i

        # Update highest since entry
        if (not np.isnan(high_i)) and (high_i > state.highest):
            state.highest = high_i

        # Check for exit: MACD bearish cross
        if macd_cross_down(i):
            # Reset state
            state.in_position = False
            state.entry_i = None
            state.highest = -np.inf
            state.entry_price = np.nan
            return (-np.inf, 2, 1)

        # Check trailing stop
        if (not np.isnan(state.highest)) and (not np.isnan(atr_i)) and (not np.isnan(close_i)):
            trail_level = state.highest - trailing_mult * atr_i
            if close_i < trail_level:
                # Reset state and close position
                state.in_position = False
                state.entry_i = None
                state.highest = -np.inf
                state.entry_price = np.nan
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
    Compute indicators required by the strategy using vectorbt.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a numpy array matching the length of the input ohlcv DataFrame.
    """
    # Validate input columns according to DATA_SCHEMA
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns or "low" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high', 'low', and 'close' columns")

    close_series = ohlcv["close"].astype(float)
    high_series = ohlcv["high"].astype(float)
    low_series = ohlcv["low"].astype(float)

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

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }
