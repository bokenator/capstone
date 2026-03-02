import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Global context state mapping (keyed by id(c)).
# Each value is a list: [highest_since_entry (float), entry_index (int)].
_CTX_STATE: Dict[int, list] = {}


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

    State is stored in a global dictionary keyed by id(c) to avoid assigning
    new attributes on the OrderContext object.
    """
    i = int(c.i)
    ctx_id = id(c)

    # Initialize per-context state if missing. Use a small list to avoid string keys
    state = _CTX_STATE.get(ctx_id)
    if state is None:
        state = [np.nan, -1]
        _CTX_STATE[ctx_id] = state

    # Safe extraction of current prices
    close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan

    # Current position size
    pos = float(getattr(c, 'position_now', 0.0))

    # If in position, update running highest using only available historical/current data
    if pos > 0:
        if np.isnan(state[0]) or state[1] < 0:
            # Initialize tracking on first bar while in position
            state[0] = high_i if not np.isnan(high_i) else close_i
            state[1] = i
        else:
            if not np.isnan(high_i):
                state[0] = max(state[0], high_i)

    # Helper functions to detect MACD crosses without raising on NaNs
    def is_macd_bull_cross(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx] > signal[idx]) and (macd[idx - 1] <= signal[idx - 1])

    def is_macd_bear_cross(idx: int) -> bool:
        if idx <= 0:
            return False
        if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
            return False
        return (macd[idx] < signal[idx]) and (macd[idx - 1] >= signal[idx - 1])

    # ENTRY logic (flat -> long)
    if pos == 0 or np.isclose(pos, 0.0):
        if is_macd_bull_cross(i):
            if not np.isnan(sma[i]) and not np.isnan(close_i):
                if close_i > float(sma[i]):
                    # Prepare trailing stop tracking on entry
                    state[0] = high_i if not np.isnan(high_i) else close_i
                    state[1] = i
                    # Use 50% of equity for the position
                    return (0.5, 2, 1)

    # EXIT logic (long -> flat)
    else:
        # 1) MACD bearish cross
        if is_macd_bear_cross(i):
            # Reset stored state
            state[0] = np.nan
            state[1] = -1
            return (-np.inf, 2, 1)

        # 2) Trailing stop based on highest_since_entry - trailing_mult * ATR
        if (not np.isnan(state[0])) and (not np.isnan(atr[i])) and (not np.isnan(close_i)):
            trailing_stop_price = float(state[0]) - float(trailing_mult) * float(atr[i])
            if close_i < trailing_stop_price:
                state[0] = np.nan
                state[1] = -1
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
    Compute MACD, ATR and SMA indicators using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("Input ohlcv DataFrame must contain 'close' and 'high' columns")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute indicators with vectorbt
    macd_res = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_res = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    sma_res = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_res.macd.values,
        'signal': macd_res.signal.values,
        'atr': atr_res.atr.values,
        'sma': sma_res.ma.values,
    }