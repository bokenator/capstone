import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple

# Global state to track (highest_price_since_entry, entry_i, entry_price) per column
_ENTRY_STATE: Dict[int, Tuple[float, int, float]] = {}


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
    Order function combining MACD crossover entries with ATR-based trailing stops.
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0)) if hasattr(c, 'col') else 0

    # Reset state at the start of a new run
    if i == 0:
        _ENTRY_STATE.clear()

    pos = float(getattr(c, 'position_now', 0.0))
    in_position = not np.isclose(pos, 0.0)

    def is_not_nan(val: Any) -> bool:
        return val is not None and not (isinstance(val, float) and np.isnan(val))

    def macd_crossed_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = macd[idx - 1], signal[idx - 1]
        a_curr, b_curr = macd[idx], signal[idx]
        if not (is_not_nan(a_prev) and is_not_nan(b_prev) and is_not_nan(a_curr) and is_not_nan(b_curr)):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def macd_crossed_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = macd[idx - 1], signal[idx - 1]
        a_curr, b_curr = macd[idx], signal[idx]
        if not (is_not_nan(a_prev) and is_not_nan(b_prev) and is_not_nan(a_curr) and is_not_nan(b_curr)):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    # Current values (with NaN handling)
    curr_close = float(close[i]) if is_not_nan(close[i]) else np.nan
    curr_high = float(high[i]) if is_not_nan(high[i]) else np.nan
    curr_sma = float(sma[i]) if is_not_nan(sma[i]) else np.nan
    curr_atr = float(atr[i]) if is_not_nan(atr[i]) else np.nan

    if not in_position:
        # Clear stale state if any
        if col in _ENTRY_STATE:
            try:
                del _ENTRY_STATE[col]
            except Exception:
                pass

        # Entry: MACD crosses up AND price above SMA
        if macd_crossed_up(i) and is_not_nan(curr_sma) and is_not_nan(curr_close):
            if curr_close > curr_sma:
                # Initialize entry state tuple: (highest, entry_i, entry_price)
                init_high = curr_high if is_not_nan(curr_high) else curr_close
                _ENTRY_STATE[col] = (float(init_high), int(i), float(curr_close))
                # Enter long with 50% of equity
                return (0.5, 2, 1)

    else:
        # Ensure entry state exists
        if col not in _ENTRY_STATE:
            init_high = curr_high if is_not_nan(curr_high) else curr_close
            _ENTRY_STATE[col] = (float(init_high), int(i), float(curr_close))

        prev_high, entry_i, entry_price = _ENTRY_STATE[col]
        # Update highest
        if is_not_nan(curr_high):
            new_high = max(prev_high, curr_high)
        else:
            new_high = prev_high
        _ENTRY_STATE[col] = (float(new_high), entry_i, entry_price)

        # Compute trailing stop level if ATR available
        trail_level = None
        if is_not_nan(new_high) and is_not_nan(curr_atr):
            trail_level = float(new_high) - float(trailing_mult) * float(curr_atr)

        # Exit conditions: MACD bearish cross OR price below trailing stop
        bad_macd = macd_crossed_down(i)
        tripped_trail = False
        if (trail_level is not None) and is_not_nan(curr_close):
            tripped_trail = (curr_close < trail_level)

        if bad_macd or tripped_trail:
            # Clear state and close position
            try:
                del _ENTRY_STATE[col]
            except Exception:
                pass
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
    Precompute indicators using vectorbt.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_sr = ohlcv['close']
    high_sr = ohlcv['high']
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Ensure float arrays
    close_arr = close_sr.astype('float64').values
    high_arr = high_sr.astype('float64').values
    low_arr = low_sr.astype('float64').values

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values.astype('float64')
    signal_arr = macd_ind.signal.values.astype('float64')

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = atr_ind.atr.values.astype('float64')

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = sma_ind.ma.values.astype('float64')

    return {
        'close': close_arr,
        'high': high_arr,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }