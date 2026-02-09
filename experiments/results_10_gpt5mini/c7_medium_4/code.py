import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Tuple

# State dictionaries to keep track of highest price since entry per column
_highest_since_entry: Dict[int, float] = {}
_entry_index: Dict[int, int] = {}


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

    This is a regular Python function (NO NUMBA).
    """
    i = int(c.i)
    pos = float(c.position_now)
    col = int(getattr(c, 'col', 0))

    # Safely get current values
    try:
        price = float(close[i])
    except Exception:
        return (np.nan, 0, 0)

    # If any of the required indicators for decision are NaN, do nothing
    # (this avoids making decisions during warmup)
    def is_nan(*vals):
        for v in vals:
            if v is None:
                return True
            try:
                if np.isnan(v):
                    return True
            except Exception:
                pass
        return False

    macd_now = macd[i] if i < len(macd) else np.nan
    sig_now = signal[i] if i < len(signal) else np.nan
    sma_now = sma[i] if i < len(sma) else np.nan
    high_now = high[i] if i < len(high) else price
    atr_now = atr[i] if i < len(atr) else np.nan

    # Entry logic: no position currently
    if pos == 0.0:
        # Need previous values to detect crossover
        if i == 0:
            return (np.nan, 0, 0)

        macd_prev = macd[i - 1]
        sig_prev = signal[i - 1]

        # Ensure indicators available
        if is_nan(macd_prev, sig_prev, macd_now, sig_now, sma_now):
            return (np.nan, 0, 0)

        macd_cross_up = (macd_prev <= sig_prev) and (macd_now > sig_now)
        price_above_sma = price > sma_now

        if macd_cross_up and price_above_sma:
            # Record highest price since (planned) entry
            _highest_since_entry[col] = high_now if not np.isnan(high_now) else price
            _entry_index[col] = i
            # Enter long with 100% equity
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # Position exists - update highest and check exits
    # Initialize highest if missing (defensive)
    prev_high = _highest_since_entry.get(col, high_now if not np.isnan(high_now) else price)
    # Update with current high
    try:
        if not np.isnan(high_now):
            cur_high = max(prev_high, float(high_now))
        else:
            cur_high = prev_high
    except Exception:
        cur_high = prev_high

    _highest_since_entry[col] = cur_high

    # Compute trailing stop
    trailing_stop = np.nan
    if not is_nan(cur_high, atr_now):
        trailing_stop = cur_high - float(trailing_mult) * float(atr_now)

    # Check MACD bearish cross
    if i == 0:
        macd_prev = np.nan
        sig_prev = np.nan
    else:
        macd_prev = macd[i - 1]
        sig_prev = signal[i - 1]

    macd_bearish = False
    if not is_nan(macd_prev, sig_prev, macd_now, sig_now):
        macd_bearish = (macd_prev >= sig_prev) and (macd_now < sig_now)

    price_below_trailing = False
    if not is_nan(price, trailing_stop):
        price_below_trailing = price < trailing_stop

    if macd_bearish or price_below_trailing:
        # Clear state for this column
        if col in _highest_since_entry:
            del _highest_since_entry[col]
        if col in _entry_index:
            del _entry_index[col]
        # Close entire long position
        return (-np.inf, 2, 1)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators. Use vectorbt indicator classes.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are np.ndarray of same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close_s = ohlcv['close'].astype(float)
    high_s = ohlcv['high'].astype(float)

    # low is optional per DATA_SCHEMA - fallback to close if missing
    if 'low' in ohlcv.columns:
        low_s = ohlcv['low'].astype(float)
    else:
        low_s = close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_arr = atr_ind.atr.values

    # Compute SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        'close': close_s.values,
        'high': high_s.values,
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }