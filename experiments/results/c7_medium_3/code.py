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

    This is a regular Python function (NO NUMBA).
    Returns vbt order objects created with vbt.portfolio.nb.order_nb or
    vbt.portfolio.enums.NoOrder for no action.
    """
    i = c.i
    pos = c.position_now

    # Safety: ensure indices are within bounds
    n = len(close)
    if i < 0 or i >= n:
        return vbt.portfolio.enums.NoOrder

    # Helper to safely check finite values
    def is_finite(idx, arr):
        try:
            return np.isfinite(arr[idx])
        except Exception:
            return False

    # Detect MACD cross up at index k
    def cross_up_at(k: int) -> bool:
        if not (is_finite(k, macd) and is_finite(k, signal)):
            return False
        # Must be above SMA too for an entry
        if not (is_finite(k, sma) and is_finite(k, close)):
            return False
        if not (macd[k] > signal[k]):
            return False
        if k == 0:
            return False
        if is_finite(k - 1, macd) and is_finite(k - 1, signal):
            return (macd[k - 1] <= signal[k - 1])
        return False

    # Detect MACD cross down at index k
    def cross_down_at(k: int) -> bool:
        if not (is_finite(k, macd) and is_finite(k, signal)):
            return False
        if not (macd[k] < signal[k]):
            return False
        if k == 0:
            return False
        if is_finite(k - 1, macd) and is_finite(k - 1, signal):
            return (macd[k - 1] >= signal[k - 1])
        return False

    # Current bar close
    try:
        close_i = close[i]
    except Exception:
        close_i = np.nan

    # 1) Entry logic (long-only)
    enter_now = False
    if pos == 0:
        if cross_up_at(i):
            # Trend filter: price above SMA
            if is_finite(i, sma) and is_finite(i, close) and (close_i > sma[i]):
                enter_now = True

        if enter_now:
            # Use 50% of equity to enter long
            return vbt.portfolio.nb.order_nb(
                np.inf,
                0.5,
                vbt.portfolio.enums.SizeType.Percent,
                vbt.portfolio.enums.Direction.LongOnly,
            )

    # 2) Exit logic (if in position)
    if pos != 0:
        # 2a) MACD bearish cross -> exit immediately
        if cross_down_at(i):
            return vbt.portfolio.nb.order_nb(
                np.inf,
                -np.inf,
                vbt.portfolio.enums.SizeType.Percent,
                vbt.portfolio.enums.Direction.LongOnly,
            )

        # 2b) Trailing stop based on highest price since entry
        # Find the most recent entry index (where MACD crossed up and price > SMA)
        entry_idx = None
        for j in range(i, -1, -1):
            if cross_up_at(j):
                # Confirm price > SMA at that bar
                if is_finite(j, sma) and is_finite(j, close) and (close[j] > sma[j]):
                    entry_idx = j
                    break

        if entry_idx is not None:
            # Compute highest high since entry (inclusive)
            highs = []
            for h in high[entry_idx : i + 1]:
                if np.isfinite(h):
                    highs.append(float(h))
            if len(highs) > 0:
                highest_since_entry = max(highs)
            else:
                highest_since_entry = np.nan

            # Current ATR
            atr_i = atr[i] if is_finite(i, atr) else np.nan

            # Compute trailing stop level
            if np.isfinite(highest_since_entry) and np.isfinite(atr_i):
                stop_level = highest_since_entry - trailing_mult * float(atr_i)
                # If price falls below stop level -> exit
                if np.isfinite(close_i) and (close_i < stop_level):
                    return vbt.portfolio.nb.order_nb(
                        np.inf,
                        -np.inf,
                        vbt.portfolio.enums.SizeType.Percent,
                        vbt.portfolio.enums.Direction.LongOnly,
                    )

    # No action
    return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators using vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are np.ndarray of same length as input.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Required columns per DATA_SCHEMA: 'close' and 'high' are required
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' and 'high' columns")

    # Prepare series (ensure float dtype)
    close_s = pd.Series(ohlcv['close'].astype(np.float64).values)
    high_s = pd.Series(ohlcv['high'].astype(np.float64).values)
    # low may be optional in schema - fall back to close if missing
    if 'low' in ohlcv.columns:
        low_s = pd.Series(ohlcv['low'].astype(np.float64).values)
    else:
        # Fallback: use close as a conservative low value
        low_s = close_s.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd.values
    signal_line = macd_ind.signal.values

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr = atr_ind.atr.values

    # Compute SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma = sma_ind.ma.values

    # Return arrays
    return {
        'close': close_s.values,
        'high': high_s.values,
        'macd': macd_line,
        'signal': signal_line,
        'atr': atr,
        'sma': sma,
    }