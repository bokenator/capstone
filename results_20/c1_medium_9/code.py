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

    This implementation keeps lightweight Python-side state on the function
    object to track the highest price since the last entry. State is stored
    as a simple list to avoid using string keys that might be mistaken for
    DataFrame column accesses.

    State layout (list): [in_position (bool), entry_i (int or -1), highest (float)]

    Rules:
    - Entry: MACD crosses above Signal AND close > SMA
    - Exit: MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    Returns tuple (size, size_type, direction) as specified in the prompt.
    """
    i = int(c.i)
    pos = float(c.position_now)  # current position size (0.0 if flat)

    # Initialize persistent state on the function object at the start of the run
    if not hasattr(order_func, "_state") or i == 0:
        in_pos = bool(pos > 0)
        entry_idx = i if in_pos else -1
        highest = float(high[i]) if (in_pos and not np.isnan(high[i])) else -np.inf
        order_func._state = [in_pos, entry_idx, highest]

    state = order_func._state

    # Unpack state for readability
    in_position = bool(state[0])
    entry_i = int(state[1]) if state[1] is not None else -1
    highest_since = float(state[2])

    # Sync state with actual position when possible (robustness)
    if pos > 0:
        if not in_position:
            # Position exists but our state didn't know about it
            in_position = True
            entry_i = i
            highest_since = float(high[i]) if not np.isnan(high[i]) else float(close[i])
        else:
            # Update highest price since entry
            if not np.isnan(high[i]):
                highest_since = max(highest_since, float(high[i]))
    else:
        in_position = False
        entry_i = -1
        highest_since = -np.inf

    # Save back to state
    state[0] = in_position
    state[1] = entry_i
    state[2] = highest_since

    # Helper to check valid index and non-NaN values
    def valid_idx(arr: np.ndarray, idx: int) -> bool:
        return 0 <= idx < arr.shape[0] and not np.isnan(arr[idx])

    # ENTRY: only when flat
    if pos == 0:
        if i > 0:
            if (
                valid_idx(macd, i - 1)
                and valid_idx(signal, i - 1)
                and valid_idx(macd, i)
                and valid_idx(signal, i)
                and valid_idx(close, i)
                and valid_idx(sma, i)
            ):
                macd_prev = macd[i - 1]
                sig_prev = signal[i - 1]
                macd_curr = macd[i]
                sig_curr = signal[i]

                macd_cross_up = (macd_prev <= sig_prev) and (macd_curr > sig_curr)
                price_above_sma = close[i] > sma[i]

                if macd_cross_up and price_above_sma:
                    # Enter long: allocate 100% of equity
                    state[0] = True
                    state[1] = i
                    state[2] = float(high[i]) if not np.isnan(high[i]) else float(close[i])
                    return (1.0, 2, 1)

    # EXIT: when in position
    else:
        # Ensure highest includes current bar
        if not np.isnan(high[i]):
            state[2] = max(state[2], float(high[i]))

        # Compute trailing stop if ATR is available
        stop_price = None
        if valid_idx(atr, i) and state[2] != -np.inf:
            stop_price = state[2] - float(trailing_mult) * float(atr[i])

        # Condition 1: MACD crosses below signal
        macd_cross_down = False
        if i > 0 and valid_idx(macd, i - 1) and valid_idx(signal, i - 1) and valid_idx(macd, i) and valid_idx(signal, i):
            macd_cross_down = (macd[i - 1] >= signal[i - 1]) and (macd[i] < signal[i])

        # Condition 2: Price falls below trailing stop
        price_below_trail = False
        if (stop_price is not None) and valid_idx(close, i):
            price_below_trail = close[i] < stop_price

        if macd_cross_down or price_below_trail:
            # Exit entire position
            state[0] = False
            state[1] = -1
            state[2] = -np.inf
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
    Precompute indicators used by the strategy using vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input ohlcv index.
    """
    # Validate input columns
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_s = ohlcv['close']
    high_s = ohlcv['high']
    # Low is optional in DATA_SCHEMA; ATR requires low - fall back to close if missing
    low_s = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    return {
        'close': close_s.to_numpy(dtype=float),
        'high': high_s.to_numpy(dtype=float),
        'macd': macd_ind.macd.to_numpy(dtype=float),
        'signal': macd_ind.signal.to_numpy(dtype=float),
        'atr': atr_ind.atr.to_numpy(dtype=float),
        'sma': sma_ind.ma.to_numpy(dtype=float),
    }
