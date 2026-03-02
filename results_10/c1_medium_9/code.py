import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple


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
    Generate order at each bar for a MACD + ATR trailing stop strategy.

    Logic:
    - Entry (when flat): MACD crosses above Signal AND price > SMA
    - Exit (when long): MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Uses attributes on the OrderContext `c` to track pending entry/exit and the
    highest price observed since entry.

    Returns a tuple (size, size_type, direction) as required by the backtester.
    - Entry uses 100% of equity: (1.0, 2, 1)
    - Exit closes entire long position: (-np.inf, 2, 1)
    - No action: (np.nan, 0, 0)
    """
    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Safely get current values (may be nan)
    def safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return float('nan')

    close_i = safe_get(close, i)
    high_i = safe_get(high, i)
    macd_i = safe_get(macd, i)
    signal_i = safe_get(signal, i)
    atr_i = safe_get(atr, i)
    sma_i = safe_get(sma, i)

    # Helper: check for MACD cross up at index idx (requires idx>0)
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a = safe_get(macd, idx)
        b = safe_get(signal, idx)
        a_prev = safe_get(macd, idx - 1)
        b_prev = safe_get(signal, idx - 1)
        if np.isnan(a) or np.isnan(b) or np.isnan(a_prev) or np.isnan(b_prev):
            return False
        return (a > b) and (a_prev <= b_prev)

    # Helper: check for MACD cross down at index idx (requires idx>0)
    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a = safe_get(macd, idx)
        b = safe_get(signal, idx)
        a_prev = safe_get(macd, idx - 1)
        b_prev = safe_get(signal, idx - 1)
        if np.isnan(a) or np.isnan(b) or np.isnan(a_prev) or np.isnan(b_prev):
            return False
        return (a < b) and (a_prev >= b_prev)

    # If flat, clear any persistent state and check for entry
    if pos == 0.0:
        # Clear state when flat
        if hasattr(c, 'entry_idx'):
            try:
                delattr(c, 'entry_idx')
            except Exception:
                # Some context objects may not allow attribute deletion; ignore
                pass
        if hasattr(c, 'highest_since_entry'):
            try:
                delattr(c, 'highest_since_entry')
            except Exception:
                pass
        if hasattr(c, '_pending_exit_idx'):
            try:
                delattr(c, '_pending_exit_idx')
            except Exception:
                pass

        # Entry conditions: MACD cross up and price > SMA
        if macd_cross_up(i) and (not np.isnan(sma_i)) and (not np.isnan(close_i)) and (close_i > sma_i):
            # Record pending entry index so we can initialize highest_since_entry once position is opened
            try:
                c._pending_entry_idx = i
            except Exception:
                # If unable to set attribute on context, we still place the order
                pass
            # Use 100% of equity to enter (percent size_type=2)
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # If we have a position, ensure we have tracked the entry index and highest price
    # Initialize entry index from pending entry if available
    if pos > 0.0:
        if not hasattr(c, 'entry_idx'):
            # Try to set entry_idx from pending entry (the bar where we signalled entry)
            if hasattr(c, '_pending_entry_idx'):
                try:
                    c.entry_idx = int(getattr(c, '_pending_entry_idx'))
                    # Remove pending marker
                    try:
                        delattr(c, '_pending_entry_idx')
                    except Exception:
                        pass
                except Exception:
                    # If any issue, fallback to current bar as entry
                    try:
                        c.entry_idx = i
                    except Exception:
                        pass
            else:
                # Unknown entry point, set to current bar
                try:
                    c.entry_idx = i
                except Exception:
                    pass

        # Initialize highest_since_entry if not present
        if not hasattr(c, 'highest_since_entry'):
            # Compute highest from entry_idx (if available) to current
            try:
                entry_idx = int(getattr(c, 'entry_idx', i))
                if entry_idx < 0:
                    entry_idx = 0
                # Guard indices
                entry_idx = max(0, entry_idx)
                # Compute max high in range entry_idx..i inclusive
                if entry_idx <= i:
                    seg = high[entry_idx:i + 1]
                    # Handle empty or all-nan segment
                    try:
                        hmax = float(np.nanmax(seg))
                    except Exception:
                        hmax = float('nan')
                else:
                    hmax = float('nan')
                c.highest_since_entry = hmax
            except Exception:
                # Fallback
                try:
                    c.highest_since_entry = float(high_i)
                except Exception:
                    c.highest_since_entry = float('nan')

        # Update highest with current high
        try:
            if not np.isnan(high_i) and not np.isnan(getattr(c, 'highest_since_entry', np.nan)):
                c.highest_since_entry = float(max(getattr(c, 'highest_since_entry', high_i), high_i))
            elif not np.isnan(high_i):
                c.highest_since_entry = float(high_i)
        except Exception:
            # If assignment fails, ignore (best-effort)
            pass

        # Evaluate exits
        should_exit_macd = macd_cross_down(i)

        should_exit_trail = False
        # Compute trailing stop level if we have ATR and highest_since_entry
        hs = getattr(c, 'highest_since_entry', np.nan)
        if (not np.isnan(hs)) and (not np.isnan(atr_i)):
            trailing_stop_level = hs - float(trailing_mult) * float(atr_i)
            # Use close price to determine breach
            if (not np.isnan(close_i)) and (close_i < trailing_stop_level):
                should_exit_trail = True
        else:
            trailing_stop_level = float('nan')

        if should_exit_macd or should_exit_trail:
            # Mark pending exit (for clarity) and request full close
            try:
                c._pending_exit_idx = i
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
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned to the input ohlcv index.
    """
    # Validate required columns from DATA_SCHEMA
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' and 'high' columns")

    close = ohlcv['close']
    high = ohlcv['high']
    # Use low if provided, otherwise fallback to close (best-effort)
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }
