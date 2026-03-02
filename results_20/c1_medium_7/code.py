import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any


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

    This implementation uses MACD crossover for entries (MACD crosses above Signal)
    with a 50-period SMA trend filter, and ATR-based trailing stops for exits.

    Entry:
      - MACD line crosses above Signal line (current bar)
      - Close > SMA
    Exit (any):
      - MACD line crosses below Signal line
      - Close falls below (highest_since_entry - trailing_mult * ATR[current])

    Notes:
      - Uses a backward scan to approximate the most recent entry bar (the last bar
        where the entry condition held). This is needed to compute the highest price
        since the (approximate) entry for the trailing stop.
      - Returns (size, size_type, direction) where size is a fraction of equity
        (Percent) when entering (e.g., 0.95 = 95% of equity), and uses
        -np.inf with Percent to close the entire long position.
    """
    i = int(c.i)
    pos = float(c.position_now)  # current position size (0.0 if flat)

    # Safety checks for array lengths
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    def is_valid_index(idx: int) -> bool:
        return 0 <= idx < n

    def is_nan_val(x: Any) -> bool:
        try:
            return np.isnan(x)
        except Exception:
            return False

    def macd_cross_up_at(idx: int) -> bool:
        if idx <= 0 or not is_valid_index(idx):
            return False
        prev_macd = macd[idx - 1]
        prev_sig = signal[idx - 1]
        cur_macd = macd[idx]
        cur_sig = signal[idx]
        if any(is_nan_val(v) for v in (prev_macd, prev_sig, cur_macd, cur_sig)):
            return False
        return (prev_macd <= prev_sig) and (cur_macd > cur_sig)

    def macd_cross_down_at(idx: int) -> bool:
        if idx <= 0 or not is_valid_index(idx):
            return False
        prev_macd = macd[idx - 1]
        prev_sig = signal[idx - 1]
        cur_macd = macd[idx]
        cur_sig = signal[idx]
        if any(is_nan_val(v) for v in (prev_macd, prev_sig, cur_macd, cur_sig)):
            return False
        return (prev_macd >= prev_sig) and (cur_macd < cur_sig)

    def find_last_entry_index(upto: int) -> int | None:
        # Search backwards for the most recent bar where the entry condition held
        for j in range(upto, -1, -1):
            if not is_valid_index(j):
                continue
            if is_nan_val(close[j]) or is_nan_val(sma[j]):
                continue
            if close[j] <= sma[j]:
                continue
            if macd_cross_up_at(j):
                return j
        return None

    # ENTRY: only when flat
    if pos == 0.0:
        # Check entry conditions at current bar
        if macd_cross_up_at(i) and (not is_nan_val(sma[i])) and (not is_nan_val(close[i])) and (close[i] > sma[i]):
            # Enter with a percentage of equity (long only)
            # Use 95% of equity to deploy capital while keeping a small cash buffer
            return (0.95, 2, 1)

        return (np.nan, 0, 0)

    # If here, we have a long position - check exit conditions
    # 1) MACD cross down
    if macd_cross_down_at(i):
        return (-np.inf, 2, 1)  # Close entire long position

    # 2) ATR-based trailing stop: compute highest price since (approx.) entry
    entry_idx = find_last_entry_index(i)
    if entry_idx is None:
        # Fallback: if we cannot find an entry index, avoid forcing an exit (rely on MACD)
        return (np.nan, 0, 0)

    # Compute highest high since entry (inclusive)
    try:
        window_highs = high[entry_idx : i + 1]
        if len(window_highs) == 0:
            return (np.nan, 0, 0)
        highest_price = np.nanmax(window_highs)
    except Exception:
        highest_price = np.nan

    if is_nan_val(highest_price) or is_nan_val(atr[i]):
        return (np.nan, 0, 0)

    trailing_stop = float(highest_price) - float(trailing_mult) * float(atr[i])

    # If price falls below trailing stop, exit
    if (not is_nan_val(close[i])) and (close[i] < trailing_stop):
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
    Precompute MACD, ATR, and SMA indicators required by the strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays of the same length as the input ohlcv.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # Use low if available, otherwise approximate with close to allow ATR computation
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    # Return plain numpy arrays
    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }
