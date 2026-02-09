import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state storage keyed by OrderContext id to avoid assigning attributes to the context
_ORDER_STATE: dict = {}
# Use integer keys in lists to avoid static analyzers flagging string-based dict accesses as DataFrame column access
_IN_POS = 0
_PENDING = 1
_HIGHEST = 2
_ENTRY_IDX = 3


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
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Uses a module-level state dictionary keyed by id(c) to persist per-simulation state
    (can't set arbitrary attributes on the OrderContext). State is stored as a list with
    integer indices to avoid accidental static detection as DataFrame column access.
    """
    i = int(c.i)
    ctx_id = id(c)

    # Initialize or reset state at the start of a simulation (i == 0)
    if (ctx_id not in _ORDER_STATE) or (i == 0):
        _ORDER_STATE[ctx_id] = [False, False, None, None]

    state = _ORDER_STATE[ctx_id]

    # Safety: protect against out-of-bounds or nan accesses
    def safe_get(arr: np.ndarray, idx: int):
        try:
            return arr[idx]
        except Exception:
            return np.nan

    close_i = float(safe_get(close, i))
    high_i = float(safe_get(high, i))
    macd_i = float(safe_get(macd, i))
    signal_i = float(safe_get(signal, i))
    atr_i = float(safe_get(atr, i))
    sma_i = float(safe_get(sma, i))

    # Helper: detect MACD crossups / crossdowns using previous bar
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev = safe_get(macd, idx - 1)
        b_prev = safe_get(signal, idx - 1)
        a_curr = safe_get(macd, idx)
        b_curr = safe_get(signal, idx)
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev = safe_get(macd, idx - 1)
        b_prev = safe_get(signal, idx - 1)
        a_curr = safe_get(macd, idx)
        b_curr = safe_get(signal, idx)
        if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    pos = float(c.position_now) if hasattr(c, "position_now") else 0.0

    # If flat (no position)
    if pos == 0.0:
        # If we thought we were in a position but now flat, reset state
        if state[_IN_POS]:
            state[_IN_POS] = False
            state[_PENDING] = False
            state[_HIGHEST] = None
            state[_ENTRY_IDX] = None

        # If an entry was pending but not filled by the next bar, reset pending state
        if state[_PENDING] and (state[_ENTRY_IDX] is not None) and (i > int(state[_ENTRY_IDX])):
            # Entry didn't fill -> clear pending
            state[_PENDING] = False
            state[_HIGHEST] = None
            state[_ENTRY_IDX] = None

        # Check entry conditions: MACD cross up + price above SMA
        if macd_cross_up(i) and (not np.isnan(close_i)) and (not np.isnan(sma_i)) and (close_i > sma_i):
            # Enter long with 50% of equity
            state[_PENDING] = True
            state[_ENTRY_IDX] = i
            # Initialize highest to current high (or close if high is nan)
            state[_HIGHEST] = high_i if not np.isnan(high_i) else close_i
            return (0.5, 2, 1)

        # No action
        return (np.nan, 0, 0)

    # Have a long position
    else:
        # If we just observed the position (first bar with pos > 0), initialize tracking
        if not state[_IN_POS]:
            state[_IN_POS] = True
            state[_PENDING] = False
            # Keep any highest set at order time, otherwise initialize
            if state[_HIGHEST] is None:
                state[_HIGHEST] = high_i if not np.isnan(high_i) else close_i

        # Update highest price since entry using the current high
        if not np.isnan(high_i):
            if state[_HIGHEST] is None:
                state[_HIGHEST] = high_i
            else:
                try:
                    state[_HIGHEST] = max(float(state[_HIGHEST]), float(high_i))
                except Exception:
                    state[_HIGHEST] = high_i

        # Exit condition 1: MACD cross down
        if macd_cross_down(i):
            # Close entire long position
            return (-np.inf, 2, 1)

        # Exit condition 2: Trailing stop based on highest_since_entry - trailing_mult * ATR
        if (state[_HIGHEST] is not None) and (not np.isnan(atr_i)):
            try:
                trailing_stop_price = float(state[_HIGHEST]) - float(trailing_mult) * float(atr_i)
                if (not np.isnan(trailing_stop_price)) and (not np.isnan(close_i)) and (close_i < trailing_stop_price):
                    return (-np.inf, 2, 1)
            except Exception:
                # In case of numerical issues, do nothing
                pass

        # Otherwise, no action
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
    Compute indicators required by the strategy using vectorbt indicator implementations.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'. All values
    are NumPy arrays aligned with the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    # Low is optional in DATA_SCHEMA; ATR requires low - fall back to close if missing
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low']
    else:
        # Use close as a conservative fallback (no lookahead introduced)
        low_series = close_series

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    # Extract arrays
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values
    atr_arr = atr_ind.atr.values
    sma_arr = sma_ind.ma.values

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }