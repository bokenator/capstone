# Complete implementation for MACD + ATR trailing stop strategy using vectorbt
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


# Module-level state to track per-column entry highs
_ENTRY_STATE: Dict[int, Dict[str, Any]] = {}


def compute_indicators(
    ohlcv: Union[pd.DataFrame, pd.Series],
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Accepts either a DataFrame with OHLCV columns or a Series of close prices.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are NumPy arrays aligned with the input length.
    """
    # Normalize inputs
    if isinstance(ohlcv, pd.Series):
        close_s = ohlcv.astype(float)
        # If only close series provided, use it for high/low as well
        high_s = close_s
        low_s = close_s
    elif isinstance(ohlcv, pd.DataFrame):
        if "close" not in ohlcv.columns:
            raise ValueError("DataFrame must contain 'close' column")
        close_s = ohlcv["close"].astype(float)
        # Use high/low if available, otherwise fall back to close
        high_s = ohlcv["high"].astype(float) if "high" in ohlcv.columns else close_s
        low_s = ohlcv["low"].astype(float) if "low" in ohlcv.columns else close_s
    else:
        # Try to coerce numpy array-like into a Series
        close_s = pd.Series(np.asarray(ohlcv).astype(float))
        high_s = close_s
        low_s = close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(
        close_s,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )
    macd_arr = np.asarray(macd_ind.macd).ravel()
    signal_arr = np.asarray(macd_ind.signal).ravel()

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_arr = np.asarray(atr_ind.atr).ravel()

    # Compute SMA (simple moving average)
    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_arr = np.asarray(sma_ind.ma).ravel()

    # Export arrays (ensure 1D numpy arrays)
    close_arr = np.asarray(close_s).ravel()
    high_arr = np.asarray(high_s).ravel()

    # Sanity: all arrays must have the same length
    n = len(close_arr)
    for name, arr in (
        ("macd", macd_arr),
        ("signal", signal_arr),
        ("atr", atr_arr),
        ("sma", sma_arr),
        ("high", high_arr),
    ):
        if len(arr) != n:
            raise ValueError(f"Indicator {name} has mismatched length {len(arr)} vs close {n}")

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float = 2.0,
) -> Tuple[float, int, int]:
    """
    Order function implementing the strategy logic for vectorbt.Portfolio.from_order_func.

    Args:
        c: Context provided by vectorbt (has attributes like `i`, `col`, `position_now`).
        close, high, macd, signal, atr, sma: NumPy arrays of indicator values.
        trailing_mult: Multiplier for ATR when computing trailing stop.

    Returns:
        A tuple (size, size_type, direction) that will be converted to a vectorbt Order.

    Notes:
        - Long-only strategy.
        - Entry: MACD crosses above its signal AND price > 50-period SMA.
        - Exit: MACD crosses below OR price < highest_since_entry - trailing_mult * ATR.
    """
    # Identify column (support multi-column runs) and current index
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0
    idx = int(getattr(c, "i", 0))

    # Initialize per-column state if needed
    state = _ENTRY_STATE.setdefault(col, {"entry_high": -np.inf})

    # Current position (float, >0 means long)
    position_now = float(getattr(c, "position_now", 0.0))

    # Safely get current and previous indicator values (avoid IndexError)
    def safe_get(arr: np.ndarray, i: int) -> float:
        try:
            return float(arr[i])
        except Exception:
            return float("nan")

    price = safe_get(close, idx)
    high_price = safe_get(high, idx)
    atr_now = safe_get(atr, idx)
    sma_now = safe_get(sma, idx)

    macd_now = safe_get(macd, idx)
    signal_now = safe_get(signal, idx)
    macd_prev = safe_get(macd, idx - 1) if idx - 1 >= 0 else float("nan")
    signal_prev = safe_get(signal, idx - 1) if idx - 1 >= 0 else float("nan")

    # Detect MACD cross up and cross down (no lookahead)
    macd_cross_up = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_now)
        and not np.isnan(signal_now)
        and (macd_prev <= signal_prev)
        and (macd_now > signal_now)
    )

    macd_cross_down = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_now)
        and not np.isnan(signal_now)
        and (macd_prev >= signal_prev)
        and (macd_now < signal_now)
    )

    # ENTRY: only when not already long
    if position_now == 0.0:
        # Trend filter: price must be above SMA
        if macd_cross_up and (not np.isnan(sma_now)) and (price > sma_now):
            # Initialize entry high with the current high
            state["entry_high"] = high_price if not np.isnan(high_price) else price
            # Enter: target 100% long (percent = 1.0)
            return (1.0, SizeType.Percent, Direction.LongOnly)

        # Otherwise, no order
        return (np.nan, SizeType.Amount, Direction.Both)

    # If currently long, check exits and update trailing high
    if position_now > 0.0:
        # Update highest_since_entry
        if not np.isnan(high_price):
            # Only increase the entry_high; never decrease it while holding
            if state.get("entry_high", -np.inf) is None or state.get("entry_high", -np.inf) == -np.inf:
                state["entry_high"] = high_price
            else:
                state["entry_high"] = max(state.get("entry_high", high_price), high_price)

        # Trailing stop level
        trailing_stop = (
            state.get("entry_high", -np.inf) - trailing_mult * atr_now
            if (not np.isnan(state.get("entry_high", np.nan)) and not np.isnan(atr_now))
            else float("nan")
        )

        # Exit on MACD cross down or when price falls below trailing stop
        if (macd_cross_down) or (not np.isnan(trailing_stop) and not np.isnan(price) and (price < trailing_stop)):
            # Reset entry high
            state["entry_high"] = -np.inf
            # Exit: target 0% long
            return (0.0, SizeType.Percent, Direction.LongOnly)

        # Otherwise, hold
        return (np.nan, SizeType.Amount, Direction.Both)

    # Default: do nothing
    return (np.nan, SizeType.Amount, Direction.Both)


# Expose only the required functions
__all__ = ["compute_indicators", "order_func"]
