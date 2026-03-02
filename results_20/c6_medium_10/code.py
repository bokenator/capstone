# Complete implementation for MACD + ATR trailing stop strategy using vectorbt

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dict with numpy arrays: macd, signal, atr, sma, close, high.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close',...]
        macd_fast, macd_slow, macd_signal: MACD parameters
        sma_period: period for trend filter SMA
        atr_period: period for ATR

    Returns:
        Dict of numpy arrays, each 1D and matching the length and index of ohlcv.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    # ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    # SMA (using MA indicator)
    sma_ind = vbt.MA.run(close, sma_period)

    # Extract outputs as 1D numpy arrays
    macd = np.asarray(macd_ind.macd).ravel()
    signal = np.asarray(macd_ind.signal).ravel()
    atr = np.asarray(atr_ind.atr).ravel()
    sma = np.asarray(sma_ind.ma).ravel()

    # Ensure arrays have correct length
    n = len(close)
    def _ensure_len(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr).ravel()
        if arr.shape[0] != n:
            # try to reindex using the original index
            arr = pd.Series(arr, index=close.index[: arr.shape[0]]).reindex(close.index).values
        return arr

    macd = _ensure_len(macd)
    signal = _ensure_len(signal)
    atr = _ensure_len(atr)
    sma = _ensure_len(sma)

    return {
        "macd": macd,
        "signal": signal,
        "atr": atr,
        "sma": sma,
        "close": np.asarray(close).ravel(),
        "high": np.asarray(high).ravel(),
    }


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt.Portfolio.from_order_func.

    Returns a tuple (size, size_type, direction). Use np.nan for size to indicate no order.

    Strategy logic:
    - Entry when MACD crosses above Signal AND price > SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Notes:
    - Uses context object `c` to access current index (c.i) and current position (c.position_now).
    - Attempts to store small state on `c` (c._entry_idx and c._highest) to track highest price since entry.
      If attribute assignment is not allowed, falls back to computing highest from recorded pos_record if available,
      or uses a conservative rolling maximum.
    """
    # Defensive conversions
    close = np.asarray(close).ravel()
    high = np.asarray(high).ravel()
    macd = np.asarray(macd).ravel()
    signal = np.asarray(signal).ravel()
    atr = np.asarray(atr).ravel()
    sma = np.asarray(sma).ravel()

    i = int(getattr(c, "i", 0))

    # Warmup checks: if indicators are not available yet, do nothing
    if i <= 0:
        return (np.nan, 0, 2)

    # Safe element getter
    def _val(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return float("nan")

    macd_i = _val(macd, i)
    signal_i = _val(signal, i)
    macd_prev = _val(macd, i - 1)
    signal_prev = _val(signal, i - 1)
    sma_i = _val(sma, i)
    close_i = _val(close, i)
    high_i = _val(high, i)
    atr_i = _val(atr, i)

    # Helpers for cross detection
    def macd_cross_up() -> bool:
        if np.isnan(macd_i) or np.isnan(signal_i) or np.isnan(macd_prev) or np.isnan(signal_prev):
            return False
        return (macd_i > signal_i) and (macd_prev <= signal_prev)

    def macd_cross_down() -> bool:
        if np.isnan(macd_i) or np.isnan(signal_i) or np.isnan(macd_prev) or np.isnan(signal_prev):
            return False
        return (macd_i < signal_i) and (macd_prev >= signal_prev)

    # Current position size (0 if flat)
    position_now = float(getattr(c, "position_now", 0.0))

    # Constants for order tuple (avoid importing enums)
    SIZE_TYPE_TARGET_PERCENT = 5  # TargetPercent
    DIRECTION_LONG = 0  # LongOnly
    DIRECTION_BOTH = 2  # Both

    # ENTRY: only when flat
    if position_now == 0.0 or np.isclose(position_now, 0.0):
        if macd_cross_up() and (not np.isnan(sma_i)) and (close_i > sma_i):
            # Record entry index and highest on context if possible
            try:
                setattr(c, "_entry_idx", i)
                setattr(c, "_highest", float(high_i if not np.isnan(high_i) else close_i))
            except Exception:
                # ignore if context does not allow attribute assignment
                pass

            # Go long: set target percent to 100%
            return (1.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_LONG)

        return (np.nan, 0, DIRECTION_BOTH)

    # If here, we are in a position
    # Try to update highest since entry stored on context
    highest_since_entry = None
    try:
        if hasattr(c, "_highest"):
            # update stored highest
            try:
                prev_high = float(getattr(c, "_highest"))
                if not np.isnan(high_i) and high_i > prev_high:
                    setattr(c, "_highest", float(high_i))
                    highest_since_entry = float(high_i)
                else:
                    highest_since_entry = prev_high
            except Exception:
                highest_since_entry = None
    except Exception:
        highest_since_entry = None

    # If we still don't have highest, try to compute from pos_record_now entry index
    if highest_since_entry is None:
        try:
            pr = getattr(c, "pos_record_now", None)
            entry_idx = None
            if pr is not None:
                # Try multiple ways to extract entry index
                try:
                    # numpy.void with field name
                    entry_idx = int(pr["entry_idx"])  # type: ignore
                except Exception:
                    try:
                        entry_idx = int(pr.entry_idx)  # type: ignore
                    except Exception:
                        entry_idx = None

            if entry_idx is not None and entry_idx <= i:
                # compute highest from entry_idx to i
                seg = high[entry_idx : i + 1]
                if len(seg) > 0:
                    # nanmax might error if all nan
                    highest_since_entry = float(np.nanmax(seg))
        except Exception:
            highest_since_entry = None

    # Final fallback: use recent rolling max from a reasonable window (e.g., last 200 bars)
    if highest_since_entry is None:
        try:
            start = max(0, i - 200)
            seg = high[start : i + 1]
            if len(seg) > 0:
                highest_since_entry = float(np.nanmax(seg))
            else:
                highest_since_entry = float(high_i)
        except Exception:
            highest_since_entry = float(high_i)

    # Compute trailing stop price
    trailing_stop = None
    if not np.isnan(atr_i):
        trailing_stop = highest_since_entry - float(trailing_mult) * float(atr_i)

    # EXIT: MACD bearish cross OR price below trailing stop
    should_exit = False
    if macd_cross_down():
        should_exit = True

    if (trailing_stop is not None) and (not np.isnan(close_i)) and (close_i < trailing_stop):
        should_exit = True

    if should_exit:
        # Clear stored state if possible
        try:
            if hasattr(c, "_entry_idx"):
                delattr(c, "_entry_idx")
            if hasattr(c, "_highest"):
                delattr(c, "_highest")
        except Exception:
            pass

        # Close position by setting target percent to 0
        return (0.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_BOTH)

    # Otherwise, no order
    return (np.nan, 0, DIRECTION_BOTH)
