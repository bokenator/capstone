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
    trailing_mult: float,
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).
    See prompt for specification of arguments and return format.
    """
    i = int(c.i)  # current bar index
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Safety checks: if we're at the first bar, do nothing
    if i == 0:
        return (np.nan, 0, 0)

    # Helper: safe access to arrays (handle potential out-of-bounds or nans)
    def safe_get(arr: np.ndarray, idx: int):
        try:
            return float(arr[idx])
        except Exception:
            return np.nan

    # Compute MACD cross signals (using previous bar)
    macd_prev = safe_get(macd, i - 1)
    sig_prev = safe_get(signal, i - 1)
    macd_curr = safe_get(macd, i)
    sig_curr = safe_get(signal, i)

    macd_cross_up = (
        not np.isnan(macd_prev) and not np.isnan(sig_prev)
        and not np.isnan(macd_curr) and not np.isnan(sig_curr)
        and (macd_prev < sig_prev) and (macd_curr > sig_curr)
    )

    macd_cross_down = (
        not np.isnan(macd_prev) and not np.isnan(sig_prev)
        and not np.isnan(macd_curr) and not np.isnan(sig_curr)
        and (macd_prev > sig_prev) and (macd_curr < sig_curr)
    )

    close_curr = safe_get(close, i)
    sma_curr = safe_get(sma, i)
    atr_curr = safe_get(atr, i)

    # ENTRY: when flat and MACD crosses up AND price above SMA
    if pos == 0.0:
        if macd_cross_up and (not np.isnan(close_curr)) and (not np.isnan(sma_curr)) and (close_curr > sma_curr):
            # Enter using 50% of equity (percent sizing)
            return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # If we have a long position, check exits
    # 1) MACD bearish cross
    if pos > 0.0:
        if macd_cross_down:
            # Close entire long position
            return (-np.inf, 2, 1)

        # 2) ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        # Attempt to extract entry index from the position record using attribute access only
        entry_idx = None
        try:
            pos_rec = getattr(c, 'pos_record_now', None)
            if pos_rec is not None:
                # Try attribute access for common field names
                for attr_name in ('entry_idx', 'entry_index', 'entry_i', 'entry_ix', 'entry'):
                    try:
                        val = getattr(pos_rec, attr_name, None)
                    except Exception:
                        val = None
                    if val is not None:
                        try:
                            entry_idx = int(val)
                            break
                        except Exception:
                            # If it's an array-like, try to coerce
                            try:
                                entry_idx = int(np.asarray(val).item())
                                break
                            except Exception:
                                entry_idx = None
        except Exception:
            entry_idx = None

        # More fallbacks: last_lidx (last position index) or last_oidx (last order index)
        if entry_idx is None:
            try:
                last_lidx = getattr(c, 'last_lidx', None)
                if last_lidx is not None:
                    if isinstance(last_lidx, (int, np.integer)):
                        entry_idx = int(last_lidx)
                    else:
                        try:
                            # Attempt to get a scalar from array-like
                            arr = np.asarray(last_lidx)
                            if arr.size > 0:
                                entry_idx = int(arr[-1])
                        except Exception:
                            entry_idx = None
            except Exception:
                entry_idx = None

        # If we found an entry index, compute highest high since entry
        if entry_idx is not None and (0 <= entry_idx <= i):
            try:
                segment = high[entry_idx:(i + 1)]
                if len(segment) > 0:
                    highest = float(np.nanmax(segment))
                else:
                    highest = np.nan
            except Exception:
                highest = np.nan

            if (not np.isnan(highest)) and (not np.isnan(atr_curr)):
                trailing_threshold = highest - float(trailing_mult) * atr_curr
                # If price falls below threshold, exit
                if (not np.isnan(close_curr)) and (close_curr < trailing_threshold):
                    return (-np.inf, 2, 1)

    # By default, no action
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
    Precompute all indicators. Use vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are np.ndarray of same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'high' and 'close' columns")

    # For ATR, low is required by the indicator
    if 'low' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'low' column for ATR calculation")

    # Ensure we work with float arrays and preserve length/index
    close_series = ohlcv['close'].astype(float)
    high_series = ohlcv['high'].astype(float)
    low_series = ohlcv['low'].astype(float)

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA (MA)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }