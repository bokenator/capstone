import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple, Optional

# Internal state for tracking highest price since entry. These are module-level
# so they persist across consecutive calls during a single backtest run.
_entry_index: Optional[int] = None
_entry_high: float = np.nan


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
    Order function for vectorbt.from_order_func (pure Python, NO NUMBA).

    Implements MACD crossover entries filtered by a 50-period SMA and
    ATR-based trailing stops (highest_since_entry - trailing_mult * ATR).

    Returns a tuple (size, size_type, direction) as required by the
    backtest runner wrapper.
    """
    global _entry_index, _entry_high

    i: int = int(c.i)  # current bar index
    pos: float = float(c.position_now)  # current position size (0.0 if flat)

    # Reset internal state at the start of a run to avoid leaking state across
    # multiple backtests executed in the same Python process.
    if i == 0:
        _entry_index = None
        _entry_high = np.nan

    # Safely extract current values (guard against arrays shorter than i)
    # Use try/except to avoid raising in the order function (vectorbt handles errors)
    try:
        close_i = float(close[i])
    except Exception:
        close_i = np.nan

    try:
        high_i = float(high[i])
    except Exception:
        high_i = np.nan

    try:
        macd_i = float(macd[i])
    except Exception:
        macd_i = np.nan

    try:
        signal_i = float(signal[i])
    except Exception:
        signal_i = np.nan

    try:
        atr_i = float(atr[i])
    except Exception:
        atr_i = np.nan

    try:
        sma_i = float(sma[i])
    except Exception:
        sma_i = np.nan

    # Helper to check bullish MACD crossover at current bar (uses previous bar)
    def is_bull_cross(idx: int) -> bool:
        if idx < 1:
            return False
        try:
            prev_macd = float(macd[idx - 1])
            prev_signal = float(signal[idx - 1])
            cur_macd = float(macd[idx])
            cur_signal = float(signal[idx])
        except Exception:
            return False
        if np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(cur_macd) or np.isnan(cur_signal):
            return False
        return (prev_macd < prev_signal) and (cur_macd > cur_signal)

    # Helper to check bearish MACD crossover at current bar
    def is_bear_cross(idx: int) -> bool:
        if idx < 1:
            return False
        try:
            prev_macd = float(macd[idx - 1])
            prev_signal = float(signal[idx - 1])
            cur_macd = float(macd[idx])
            cur_signal = float(signal[idx])
        except Exception:
            return False
        if np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(cur_macd) or np.isnan(cur_signal):
            return False
        return (prev_macd > prev_signal) and (cur_macd < cur_signal)

    # ENTRY: only when flat and MACD bullish crossover and price > SMA
    if pos == 0.0:
        should_enter = False
        if is_bull_cross(i):
            # Trend filter: require price above SMA
            if not np.isnan(close_i) and not np.isnan(sma_i) and (close_i > sma_i):
                should_enter = True

        if should_enter:
            # Record entry state immediately so trailing stop starts tracking from
            # the entry bar. Use the high of the entry bar as the initial high.
            _entry_index = i
            _entry_high = high_i if not np.isnan(high_i) else close_i

            # Use 50% of equity for the position (Percent size type = 2)
            return (0.5, 2, 1)

        # No action
        return (np.nan, 0, 0)

    # POSITIONED: check exits
    else:
        # If we have no recorded entry index (e.g., position was present at
        # the start of the simulation), initialize it to the current bar.
        if _entry_index is None:
            _entry_index = i
            _entry_high = high_i if not np.isnan(high_i) else close_i

        # Update highest price since entry using only current high (no lookahead)
        if not np.isnan(high_i):
            if np.isnan(_entry_high):
                _entry_high = high_i
            else:
                if high_i > _entry_high:
                    _entry_high = high_i

        # Evaluate exit conditions
        bear_cross = is_bear_cross(i)

        trail_trigger = False
        if not np.isnan(_entry_high) and not np.isnan(atr_i):
            trail_level = _entry_high - (trailing_mult * atr_i)
            # Trigger when close falls strictly below the trailing level
            if not np.isnan(close_i) and (close_i < trail_level):
                trail_trigger = True

        if bear_cross or trail_trigger:
            # Reset state on exit
            _entry_index = None
            _entry_high = np.nan

            # Close entire long position using Percent size type with -inf
            # convention (wrapper/backtester interprets this as close)
            return (-np.inf, 2, 1)

        # Otherwise, no action
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
    Compute and return indicators required by the strategy.

    Returns a dictionary with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    Each value is a numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close_sr = ohlcv['close'].astype(float)
    high_sr = ohlcv['high'].astype(float)

    # 'low' is optional in the schema; if missing, fall back to close to avoid
    # computation errors. This is a conservative approach that keeps the code
    # robust for all allowed inputs while not introducing forward-looking data.
    if 'low' in ohlcv.columns:
        low_sr = ohlcv['low'].astype(float)
        # If low contains NaNs, fill those with close to remain conservative
        if low_sr.isna().any():
            low_sr = low_sr.fillna(close_sr)
    else:
        low_sr = close_sr.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR (uses ewm by default which reduces initial NaNs)
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)

    # Extract numpy arrays; use .values to preserve alignment and length
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values
    atr_arr = atr_ind.atr.values
    sma_arr = sma_ind.ma.values

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }