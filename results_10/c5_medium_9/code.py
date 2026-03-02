import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state for trailing stop tracking. Reset at the start of each run
# (when c.i == 0) to avoid cross-run contamination.
_ENTRY_IN_POSITION: bool = False
_ENTRY_HIGH: float = np.nan


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

    Entry: MACD crosses above Signal AND close > SMA
    Exit: MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)
    """
    global _ENTRY_IN_POSITION, _ENTRY_HIGH

    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Reset state at the start of a run
    if i == 0:
        _ENTRY_IN_POSITION = False
        _ENTRY_HIGH = np.nan

    # Ensure inputs are numpy arrays
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    def is_valid_value(x):
        return not (np.isnan(x) or np.isposinf(x) or np.isneginf(x))

    def macd_cross_up_at(idx: int) -> bool:
        if idx < 1:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_curr = macd[idx]
        b_curr = signal[idx]
        if not (is_valid_value(a_prev) and is_valid_value(b_prev) and is_valid_value(a_curr) and is_valid_value(b_curr)):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def macd_cross_down_at(idx: int) -> bool:
        if idx < 1:
            return False
        a_prev = macd[idx - 1]
        b_prev = signal[idx - 1]
        a_curr = macd[idx]
        b_curr = signal[idx]
        if not (is_valid_value(a_prev) and is_valid_value(b_prev) and is_valid_value(a_curr) and is_valid_value(b_curr)):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    # Entry logic
    if pos == 0.0:
        should_enter = False
        if macd_cross_up_at(i):
            if is_valid_value(close[i]) and is_valid_value(sma[i]):
                if close[i] > sma[i]:
                    should_enter = True

        if should_enter:
            _ENTRY_IN_POSITION = True
            _ENTRY_HIGH = high[i] if is_valid_value(high[i]) else close[i]
            # Buy with 50% of equity
            return (0.5, 2, 1)

    else:
        # Ensure internal state initialized if lost
        if (not _ENTRY_IN_POSITION) or np.isnan(_ENTRY_HIGH):
            _ENTRY_IN_POSITION = True
            _ENTRY_HIGH = high[i] if is_valid_value(high[i]) else close[i]

        # Update highest since entry
        if is_valid_value(high[i]):
            if np.isnan(_ENTRY_HIGH):
                _ENTRY_HIGH = high[i]
            else:
                _ENTRY_HIGH = max(_ENTRY_HIGH, high[i])

        # Compute trailing stop level if ATR available
        trailing_level = None
        if not np.isnan(_ENTRY_HIGH) and is_valid_value(atr[i]):
            trailing_level = _ENTRY_HIGH - trailing_mult * atr[i]

        # Exit checks
        exit_by_macd = macd_cross_down_at(i)
        exit_by_trailing = False
        if trailing_level is not None and is_valid_value(close[i]):
            exit_by_trailing = close[i] < trailing_level

        if exit_by_macd or exit_by_trailing:
            _ENTRY_IN_POSITION = False
            _ENTRY_HIGH = np.nan
            return (-np.inf, 2, 1)

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' and 'close' columns")

    close_s = ohlcv['close'].astype(float)
    high_s = ohlcv['high'].astype(float)
    low_s = ohlcv['low'].astype(float) if 'low' in ohlcv.columns else close_s
    low_s = low_s.astype(float)

    macd_ind = vbt.MACD.run(close_s, fast_window=int(macd_fast), slow_window=int(macd_slow), signal_window=int(macd_signal))
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=int(atr_period))
    sma_ind = vbt.MA.run(close_s, window=int(sma_period))

    return {
        'close': close_s.to_numpy(),
        'high': high_s.to_numpy(),
        'macd': macd_ind.macd.to_numpy(),
        'signal': macd_ind.signal.to_numpy(),
        'atr': atr_ind.atr.to_numpy(),
        'sma': sma_ind.ma.to_numpy(),
    }