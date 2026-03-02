import numpy as np
import pandas as pd
import vectorbt as vbt


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

    This implementation combines MACD crossover entries with an ATR-based
    trailing stop. It's long-only and single-asset.

    Position sizing: buys use 50% of equity (size_type=2, size=0.5).

    State is stored on the function object as attributes to avoid using
    DataFrame-like string indexing. The stored state tracks the highest
    price since entry (required for the trailing stop) and the previous
    position size.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent state on the function object (persists across calls)
    if not hasattr(order_func, "_entry_highest"):
        # Highest high seen since the position was opened
        order_func._entry_highest = np.nan
    if not hasattr(order_func, "_prev_pos"):
        order_func._prev_pos = 0.0

    entry_highest = float(order_func._entry_highest) if not np.isnan(order_func._entry_highest) else np.nan

    # Safety: ensure index is valid
    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Helper: detect MACD cross up/down using previous bar
    def _macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = macd[idx - 1], signal[idx - 1]
        a1, b1 = macd[idx], signal[idx]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 <= b0) and (a1 > b1)

    def _macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a0, b0 = macd[idx - 1], signal[idx - 1]
        a1, b1 = macd[idx], signal[idx]
        if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
            return False
        return (a0 >= b0) and (a1 < b1)

    # If we are currently in a long position, update the highest price since entry
    if pos > 0:
        # Initialize entry_highest on the first bar where pos > 0
        if np.isnan(entry_highest):
            # Prefer high; fallback to close if high is nan
            val = high[i] if not np.isnan(high[i]) else close[i]
            entry_highest = float(val) if not np.isnan(val) else np.nan
        else:
            # Update running maximum with current high
            if not np.isnan(high[i]):
                entry_highest = float(max(entry_highest, high[i]))

        # Persist updated value back to function attribute
        order_func._entry_highest = entry_highest

    # ENTRY: No position and MACD crosses above signal while price > SMA
    if pos == 0.0:
        if _macd_cross_up(i):
            if not np.isnan(sma[i]) and not np.isnan(close[i]) and close[i] > sma[i]:
                # Enter long: use 50% of equity
                # Do NOT pre-initialize entry_highest here; it will be set once the
                # position appears (pos > 0) on subsequent bars.
                order_func._prev_pos = pos
                return (0.5, 2, 1)

    # EXIT: If in position, check MACD cross down OR ATR trailing stop breach
    if pos > 0:
        # MACD cross below signal
        if _macd_cross_down(i):
            # Close entire position
            order_func._entry_highest = np.nan
            order_func._prev_pos = pos
            return (-np.inf, 2, 1)

        # Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        if not np.isnan(entry_highest) and not np.isnan(atr[i]) and not np.isnan(close[i]):
            trailing_level = entry_highest - float(trailing_mult) * float(atr[i])
            # Only exit if trailing level is a valid number
            if not np.isnan(trailing_level) and close[i] < trailing_level:
                order_func._entry_highest = np.nan
                order_func._prev_pos = pos
                return (-np.inf, 2, 1)

    # No actionable signal
    order_func._prev_pos = pos
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input ohlcv index.
    """
    # Validate required columns according to DATA_SCHEMA
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain a 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain a 'high' column")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]

    # low may be missing per DATA_SCHEMA, provide fallback to close if absent
    if "low" in ohlcv.columns:
        low_series = ohlcv["low"]
    else:
        low_series = ohlcv["close"]

    # Compute MACD
    macd_ind = vbt.MACD.run(
        close_series,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values.astype(float),
        "high": high_series.values.astype(float),
        "macd": macd_ind.macd.values.astype(float),
        "signal": macd_ind.signal.values.astype(float),
        "atr": atr_ind.atr.values.astype(float),
        "sma": sma_ind.ma.values.astype(float),
    }
