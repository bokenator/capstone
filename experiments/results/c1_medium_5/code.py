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
    trailing_mult: float,
) -> tuple:
    """
    Order function using the OrderContext 'c' for transient state storage.

    Returns tuple (size, size_type, direction) where:
    - size: float (np.nan for no action, positive to buy, negative to sell)
    - size_type: int (0=Amount, 1=Value, 2=Percent)
    - direction: int (0=Both, 1=LongOnly, 2=ShortOnly)

    Entry: MACD cross up AND close > SMA -> buy with 100% of equity (1.0, 2, 1)
    Exit: MACD cross down OR close < (highest_since_entry - trailing_mult * ATR) -> close (-np.inf, 2, 1)
    """
    i = int(c.i)
    pos = float(c.position_now) if c.position_now is not None else 0.0

    # Use the order context 'c' to store state across calls
    if not hasattr(c, "_pending_entry"):
        c._pending_entry = None
        c._entry_index = None
        c._highest_since_entry = None

    def is_finite(x: Any) -> bool:
        try:
            return np.isfinite(x)
        except Exception:
            return False

    # If flat, reset confirmed entry/highest
    if pos == 0 or pos == 0.0:
        c._entry_index = None
        c._highest_since_entry = None

        # Entry conditions
        can_check_cross = i > 0 and is_finite(macd[i]) and is_finite(signal[i]) and is_finite(macd[i - 1]) and is_finite(signal[i - 1])
        if can_check_cross and is_finite(sma[i]) and is_finite(close[i]):
            macd_prev = float(macd[i - 1])
            sig_prev = float(signal[i - 1])
            macd_curr = float(macd[i])
            sig_curr = float(signal[i])
            crossed_up = (macd_prev < sig_prev) and (macd_curr > sig_curr)

            if crossed_up and (float(close[i]) > float(sma[i])):
                c._pending_entry = i
                return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # Long position is open
    if pos > 0:
        # Confirm the entry if pending
        if c._entry_index is None:
            if c._pending_entry is not None:
                c._entry_index = int(c._pending_entry)
            else:
                c._entry_index = i

            # initialize highest price since entry
            try:
                c._highest_since_entry = float(high[c._entry_index])
            except Exception:
                c._highest_since_entry = float(high[i])

        # Update highest since entry
        if is_finite(high[i]):
            if c._highest_since_entry is None:
                c._highest_since_entry = float(high[i])
            else:
                try:
                    c._highest_since_entry = max(float(c._highest_since_entry), float(high[i]))
                except Exception:
                    c._highest_since_entry = float(high[i])

        # MACD cross down
        exit_macd = False
        if i > 0 and is_finite(macd[i]) and is_finite(signal[i]) and is_finite(macd[i - 1]) and is_finite(signal[i - 1]):
            macd_prev = float(macd[i - 1])
            sig_prev = float(signal[i - 1])
            macd_curr = float(macd[i])
            sig_curr = float(signal[i])
            if (macd_prev > sig_prev) and (macd_curr < sig_curr):
                exit_macd = True

        # Trailing stop
        exit_trail = False
        if c._highest_since_entry is not None and is_finite(atr[i]) and is_finite(close[i]):
            trail_level = float(c._highest_since_entry) - float(trailing_mult) * float(atr[i])
            if float(close[i]) < trail_level:
                exit_trail = True

        if exit_macd or exit_trail:
            # clear pending/entry state (reset when flat)
            c._pending_entry = None
            c._entry_index = None
            c._highest_since_entry = None
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)

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
    Precompute indicators using vectorbt wrappers: MACD, ATR and SMA.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    # Ensure low exists for ATR
    if 'low' not in ohlcv.columns:
        low_series = ohlcv['close'].copy()
    else:
        low_series = ohlcv['low']

    close_series = ohlcv['close']
    high_series = ohlcv['high']

    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        'close': close_series.values.astype(float),
        'high': high_series.values.astype(float),
        'macd': macd_ind.macd.values.astype(float),
        'signal': macd_ind.signal.values.astype(float),
        'atr': atr_ind.atr.values.astype(float),
        'sma': sma_ind.ma.values.astype(float),
    }
