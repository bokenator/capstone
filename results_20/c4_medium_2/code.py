import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict


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
    Order function combining MACD crossover entries with ATR-based trailing stops.

    See prompt for full specification. This implementation:
    - Enters long when MACD crosses above Signal and price > SMA
    - Exits when MACD crosses below Signal OR price falls below
      (highest_since_entry - trailing_mult * ATR)

    Returns tuple (size, size_type, direction) as required by the runner.
    Uses Percent sizing for entries (50% of equity) and closes with -inf Percent.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safety checks for index bounds
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper to determine finite values
    def is_finite(x: Any) -> bool:
        try:
            return np.isfinite(x)
        except Exception:
            return False

    # Helper crossover checks (require previous bar)
    def cross_up(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx == 0:
            return False
        if not (is_finite(a[idx]) and is_finite(b[idx]) and is_finite(a[idx-1]) and is_finite(b[idx-1])):
            return False
        return (a[idx-1] <= b[idx-1]) and (a[idx] > b[idx])

    def cross_down(a: np.ndarray, b: np.ndarray, idx: int) -> bool:
        if idx == 0:
            return False
        if not (is_finite(a[idx]) and is_finite(b[idx]) and is_finite(a[idx-1]) and is_finite(b[idx-1])):
            return False
        return (a[idx-1] >= b[idx-1]) and (a[idx] < b[idx])

    # Compute signals
    macd_cross_up = cross_up(macd, signal, i)
    macd_cross_down = cross_down(macd, signal, i)

    price = float(close[i]) if is_finite(close[i]) else np.nan
    curr_sma = float(sma[i]) if is_finite(sma[i]) else np.nan
    curr_atr = float(atr[i]) if is_finite(atr[i]) else np.nan

    # No position: check entry
    if pos == 0.0:
        # Entry conditions must all be true
        can_enter = macd_cross_up and is_finite(price) and is_finite(curr_sma) and (price > curr_sma)
        if can_enter:
            # Use 50% of equity to enter (size_type=2 => percent)
            return (0.5, 2, 1)
        return (np.nan, 0, 0)

    # Have a long position: check for exits
    else:
        # 1) MACD cross down -> exit
        if macd_cross_down:
            # Close entire long position
            return (-np.inf, 2, 1)

        # 2) Trailing stop based on highest price since entry
        # Try to obtain entry index from position record (pos_record_now)
        init_i = None
        try:
            rec = getattr(c, 'pos_record_now', None)
            if rec is not None:
                # numpy.void record: try common field names
                try:
                    names = rec.dtype.names  # type: ignore
                except Exception:
                    names = None

                if names:
                    for field in ('init_i', 'init_idx', 'entry_i', 'init_index', 'entry_idx'):
                        if field in names:
                            try:
                                init_i = int(rec[field])  # type: ignore
                                break
                            except Exception:
                                init_i = None
                else:
                    # Fallback to attribute access if available
                    for attr in ('init_i', 'init_idx', 'entry_i', 'entry_idx'):
                        if hasattr(rec, attr):
                            try:
                                init_i = int(getattr(rec, attr))
                                break
                            except Exception:
                                init_i = None
        except Exception:
            init_i = None

        # Compute highest since entry
        highest_since_entry = None
        try:
            if init_i is not None and 0 <= init_i <= i:
                segment = high[init_i:i+1]
                # Use nanmax to be robust to NaNs
                highest_since_entry = float(np.nanmax(segment)) if len(segment) > 0 else None
        except Exception:
            highest_since_entry = None

        # If we couldn't read entry index or it's invalid, fall back to using the highest
        # seen since the last N bars (conservative fallback: use current high)
        if highest_since_entry is None or not is_finite(highest_since_entry):
            try:
                highest_since_entry = float(high[i]) if is_finite(high[i]) else None
            except Exception:
                highest_since_entry = None

        # Only evaluate trailing stop if we have a valid ATR and highest price
        if highest_since_entry is not None and is_finite(curr_atr):
            trail_level = highest_since_entry - float(trailing_mult) * curr_atr
            # If price falls below trail_level -> exit
            if is_finite(price) and price < trail_level:
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
    Compute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are np.ndarray aligned with the input ohlcv index.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # Low is optional in DATA_SCHEMA; if missing, fall back to close (conservative)
    low = ohlcv['low'] if 'low' in ohlcv.columns else close

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR (uses high, low, close). ATR.run returns object with .atr
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