import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy and return them as numpy arrays.

    Returns a dict with keys: macd, signal, atr, sma, close, high (all numpy arrays).
    """
    # Basic validation
    required_cols = {"close", "high", "low"}
    missing = required_cols - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD (EMA-based)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # ATR (Wilder's smoothing via ewm for stable values)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(max(1, atr_period)), adjust=False).mean()

    # SMA (trend filter) - keep NaNs for warmup periods
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # Return numpy arrays to avoid pandas label-based indexing issues inside order wrapper
    return {
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr.values,
        "sma": sma.values,
        "close": close.values,
        "high": high.values,
    }


def order_func(
    c: Any,
    close: Any,
    high: Any,
    macd: Any,
    signal: Any,
    atr: Any,
    sma: Any,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function compatible with vectorbt.Portfolio.from_order_func.

    Strategy:
    - Entry when MACD crosses above Signal AND price > SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Implementation note:
    The function replays the strategy from the start up to c.i using only data up to that point
    to ensure determinism and no lookahead.

    Orders are returned as (size, size_type, direction).
    We use Amount (size_type=0) and rely on positive size for buy and negative size for sell.
    """
    # Convert to numpy arrays for fast positional access
    close_a = np.asarray(close)
    high_a = np.asarray(high)
    macd_a = np.asarray(macd)
    signal_a = np.asarray(signal)
    atr_a = np.asarray(atr)
    sma_a = np.asarray(sma)

    # Current time index
    try:
        i = int(getattr(c, "i"))
    except Exception:
        # Fallback to last index if not provided (should not happen in normal run)
        i = len(close_a) - 1

    if i < 0:
        return (np.nan, 0, 0)

    n = len(close_a)
    if i >= n:
        # Defensive: index out of bounds
        return (np.nan, 0, 0)

    # Simulate from start to current index to determine position state at each bar
    pos_history = []
    in_pos = False
    highest_since_entry = np.nan

    # Helper to check finite number
    is_finite = np.isfinite

    for t in range(0, i + 1):
        # Values at t (use safe indexing)
        macd_t = macd_a[t]
        macd_prev = macd_a[t - 1] if t >= 1 else np.nan
        signal_t = signal_a[t]
        signal_prev = signal_a[t - 1] if t >= 1 else np.nan
        close_t = close_a[t]
        sma_t = sma_a[t]
        atr_t = atr_a[t]
        high_t = high_a[t]

        if not in_pos:
            # Check MACD bullish crossover and trend filter
            enter = False
            if (
                t >= 1
                and is_finite(macd_prev)
                and is_finite(signal_prev)
                and is_finite(macd_t)
                and is_finite(signal_t)
                and is_finite(sma_t)
                and is_finite(close_t)
            ):
                if (macd_prev < signal_prev) and (macd_t > signal_t) and (close_t > sma_t):
                    enter = True

            if enter:
                in_pos = True
                # initialize highest_since_entry with current high (fallback to close)
                highest_since_entry = high_t if is_finite(high_t) else close_t
        else:
            # Update highest price since entry
            if is_finite(high_t):
                if (not is_finite(highest_since_entry)) or (high_t > highest_since_entry):
                    highest_since_entry = high_t

            # Check MACD bearish crossover
            exit_flag = False
            if (
                t >= 1
                and is_finite(macd_prev)
                and is_finite(signal_prev)
                and is_finite(macd_t)
                and is_finite(signal_t)
            ):
                if (macd_prev > signal_prev) and (macd_t < signal_t):
                    exit_flag = True

            # Check trailing stop: price falls below highest_since_entry - trailing_mult * ATR
            if (not exit_flag) and is_finite(atr_t) and is_finite(close_t) and is_finite(highest_since_entry):
                threshold = highest_since_entry - float(trailing_mult) * atr_t
                if close_t < threshold:
                    exit_flag = True

            if exit_flag:
                in_pos = False
                highest_since_entry = np.nan

        pos_history.append(in_pos)

    # Determine if position changed at the current bar (generate an order only on change)
    curr_pos = pos_history[-1]
    prev_pos = pos_history[-2] if i >= 1 else False

    # Use Amount size type and Both direction for simple buy/sell semantics
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_BOTH = 0

    if (not prev_pos) and curr_pos:
        # Enter: buy 1 unit
        return (1.0, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    if prev_pos and (not curr_pos):
        # Exit: sell 1 unit (use negative amount to indicate selling)
        return (-1.0, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # No action
    return (np.nan, 0, 0)
