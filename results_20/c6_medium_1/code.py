# Complete implementation for the MACD + ATR trailing stop strategy
# Exports: compute_indicators, order_func

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

# Import enums for order tuples
from vectorbt.portfolio.enums import SizeType, Direction


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute technical indicators required by the strategy.

    Returns a dictionary with numpy arrays for keys:
      - 'close', 'high', 'macd', 'signal', 'atr', 'sma'

    Parameters mirror those used by the backtest runner.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise KeyError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # MACD using vectorbt
    macd_ind = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR using vectorbt
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # SMA using pandas (simple and clear)
    sma_series = close.rolling(window=sma_period, min_periods=1).mean()

    # Convert outputs to 1D numpy arrays
    # macd_ind.macd and macd_ind.signal may be pandas Series or DataFrame depending on input
    macd_arr = np.asarray(macd_ind.macd).ravel()
    signal_arr = np.asarray(macd_ind.signal).ravel()
    atr_arr = np.asarray(atr_ind.atr).ravel()
    sma_arr = np.asarray(sma_series).ravel()
    close_arr = np.asarray(close).ravel()
    high_arr = np.asarray(high).ravel()

    # Ensure all arrays have same length
    n = len(close_arr)
    for name, arr in ("macd", macd_arr), ("signal", signal_arr), ("atr", atr_arr), ("sma", sma_arr), ("high", high_arr):
        if len(arr) != n:
            # try to reshape or trim/pad if needed
            raise ValueError(f"Computed {name} has length {len(arr)} but expected {n}")

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
) -> Tuple[float, Any, Any]:
    """
    Order function used by vectorbt.Portfolio.from_order_func (non-numba).

    Returns a tuple (size, size_type, direction). If no action is desired,
    return (np.nan, SizeType.Amount, Direction.Both) so the runner treats it as no-order.

    Logic:
      - Entry: MACD crosses above Signal AND price > SMA
              -> enter by targeting 100% portfolio (SizeType.TargetPercent, size=1.0)
      - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
              -> exit by targeting 0% portfolio (SizeType.TargetPercent, size=0.0)

    The function maintains minimal internal state on the function object to track
    highest price since entry per column. State is reset when the simulation starts (c.i == 0).
    """
    # Initialize/reset state on first call (or if not present)
    idx = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0

    if not hasattr(order_func, "_state") or idx == 0:
        # state: highest price per column while in a position, and a pending entry flag
        order_func._state = {
            "highest": {},  # col -> float
            "pending_entry": {},  # col -> bool
        }

    state = order_func._state
    highest = state["highest"]
    pending_entry = state["pending_entry"]

    # Safely access current values, guarding NaNs
    def safe_get(arr: np.ndarray, i: int) -> float:
        if i < 0 or i >= len(arr):
            return np.nan
        v = arr[i]
        try:
            # numpy scalar to python float if possible
            return float(v)
        except Exception:
            return float(np.nan)

    pos_now = float(getattr(c, "position_now", 0.0))

    # MACD crossover detection (uses only past and current bars)
    cross_up = False
    cross_down = False
    if idx > 0 and idx < len(macd):
        prev_macd = safe_get(macd, idx - 1)
        prev_signal = safe_get(signal, idx - 1)
        curr_macd = safe_get(macd, idx)
        curr_signal = safe_get(signal, idx)

        if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(curr_macd) or np.isnan(curr_signal)):
            cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

    # Price and SMA
    price = safe_get(close, idx)
    sma_val = safe_get(sma, idx)
    price_above_sma = False
    if not (np.isnan(price) or np.isnan(sma_val)):
        price_above_sma = price > sma_val

    # If we are in position, ensure highest is tracked and check trailing stop / MACD exit
    if pos_now > 0:
        # Initialize highest if missing (this can happen if position was opened externally)
        if col not in highest:
            highest[col] = safe_get(high, idx)

        # Update highest with current high
        cur_high = safe_get(high, idx)
        if not np.isnan(cur_high):
            if np.isnan(highest[col]):
                highest[col] = cur_high
            else:
                highest[col] = max(highest[col], cur_high)

        # Compute trailing stop level
        atr_val = safe_get(atr, idx)
        trailing_level = highest[col] - float(trailing_mult) * (0.0 if np.isnan(atr_val) else atr_val)

        # Exit on MACD bearish cross
        if cross_down:
            # Target 0% to close long position
            # Clear highest state for this column
            highest.pop(col, None)
            pending_entry.pop(col, None)
            return (0.0, SizeType.TargetPercent, Direction.LongOnly)

        # Exit on trailing stop breach (price < trailing_level)
        if not np.isnan(price) and not np.isnan(trailing_level) and price < trailing_level:
            highest.pop(col, None)
            pending_entry.pop(col, None)
            return (0.0, SizeType.TargetPercent, Direction.LongOnly)

        # Otherwise, no action while in position
        return (np.nan, SizeType.Amount, Direction.Both)

    # Not in position: handle entry logic
    # If previously marked pending entry for this column but position still not opened, keep pending
    # (no action) — we still rely on macd cross + sma to trigger
    if cross_up and price_above_sma and pos_now == 0:
        # Set highest for the new entry to current high (helps include entry bar)
        he = safe_get(high, idx)
        highest[col] = he if not np.isnan(he) else price
        pending_entry[col] = True
        # Enter long by targeting 100% allocation
        return (1.0, SizeType.TargetPercent, Direction.LongOnly)

    # No order
    return (np.nan, SizeType.Amount, Direction.Both)
