"""
MACD + ATR trailing stop strategy using vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- compute_indicators returns numpy arrays for macd, signal, atr, sma and also passes through close/high arrays
- order_func is a plain Python function (no numba) that keeps per-column state on the function object
  to track highest price since entry for trailing stop calculation.

This file is intended to be validated and run by the provided backtest harness.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import vectorbt as vbt


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert pandas/array-like indicator output to 1D numpy array.

    Works with pandas.Series, pandas.DataFrame (single column), numpy arrays.
    """
    arr = np.asarray(x)
    # If it's a (N,1) array (e.g. DataFrame with single column), squeeze to 1D
    if arr.ndim > 1:
        # If second dimension is 1, squeeze, otherwise attempt to take first column
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            # Multiple columns: try to squeeze the first column (shouldn't happen for single-asset backtest)
            arr = arr[:, 0]
    return arr.astype(float)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute MACD, ATR and SMA indicators used by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close',...]
        macd_fast, macd_slow, macd_signal: MACD parameters
        sma_period: period for the trend filter SMA
        atr_period: ATR period

    Returns:
        dict with keys:
            - 'macd': MACD line as 1D numpy array
            - 'signal': Signal line as 1D numpy array
            - 'atr': ATR as 1D numpy array
            - 'sma': 50-period SMA as 1D numpy array
            - 'close': close price array
            - 'high': high price array
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Extract series (raise helpful error if missing)
    for col in ["close", "high", "low"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv is missing required column: {col}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD
    macd_res = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )
    macd_arr = _to_1d_array(macd_res.macd)
    signal_arr = _to_1d_array(macd_res.signal)

    # ATR
    atr_res = vbt.ATR.run(high, low, close, window=atr_period)
    atr_arr = _to_1d_array(atr_res.atr)

    # SMA (simple moving average implemented via MA with ewm=False)
    ma_res = vbt.MA.run(close, window=sma_period, ewm=False)
    sma_arr = _to_1d_array(ma_res.ma)

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": _to_1d_array(close.values),
        "high": _to_1d_array(high.values),
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
    """Order function implementing MACD crossover entries with ATR-based trailing stops.

    This function is intentionally pure Python (no numba). It keeps simple per-column state
    on the function object to remember the highest price seen since an entry.

    Args:
        c: vectorbt OrderContext-like object passed by Portfolio.from_order_func (has .i and .col and .position_now)
        close, high, macd, signal, atr, sma: 1D numpy arrays (same length)
        trailing_mult: multiplier for ATR to compute trailing stop (e.g., 2.0)

    Returns:
        tuple (size, size_type, direction)
        - size: float (np.nan for no order)
        - size_type: int (uses vectorbt SizeType enum integer values; Percent=2, TargetPercent=5)
        - direction: int (uses vectorbt Direction enum integer values; LongOnly=0, Both=2)

    Behavior:
        - Entry (open long): MACD crosses above signal AND close > SMA -> buy 100% (size_type=Percent, size=1.0)
        - Exit: MACD crosses below signal OR close < (highest_since_entry - trailing_mult * ATR)
          -> set target percent to 0 (size_type=TargetPercent, size=0.0)

    Notes:
        - The function stores per-column state in order_func._state[col]
        - We avoid importing vectorbt enums here to stay numba-free; we use integer constants
          consistent with vectorbt's enums (documented in vectorbt):
            SizeType.Percent == 2
            SizeType.TargetPercent == 5
            Direction.LongOnly == 0
            Direction.Both == 2
    """
    # Initialize internal state mapping (per column)
    if not hasattr(order_func, "_state"):
        order_func._state = {}

    # Determine column key (fallback to 0)
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0
    state = order_func._state.setdefault(
        col,
        {
            "in_position": False,
            "entry_idx": None,
            "highest_price": -np.inf,
        },
    )

    # Current bar index within arrays
    i = int(c.i) if hasattr(c, "i") else 0

    # Safe access helpers
    def _safe_get(arr: np.ndarray, idx: int) -> float:
        if idx < 0 or idx >= arr.shape[0]:
            return float("nan")
        val = arr[idx]
        try:
            return float(val)
        except Exception:
            return float("nan")

    close_i = _safe_get(close, i)
    high_i = _safe_get(high, i)
    macd_i = _safe_get(macd, i)
    signal_i = _safe_get(signal, i)
    atr_i = _safe_get(atr, i)
    sma_i = _safe_get(sma, i)

    prev_macd = _safe_get(macd, i - 1) if i > 0 else float("nan")
    prev_signal = _safe_get(signal, i - 1) if i > 0 else float("nan")

    # Crossing detection (requires previous values)
    macd_cross_up = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and prev_macd <= prev_signal
        and macd_i > signal_i
    )
    macd_cross_down = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and prev_macd >= prev_signal
        and macd_i < signal_i
    )

    # Use context position if available (preferred), else fallback to our internal state
    in_position_ctx = False
    if hasattr(c, "position_now"):
        try:
            in_position_ctx = float(c.position_now) > 0
        except Exception:
            in_position_ctx = False

    in_position = in_position_ctx or state["in_position"]

    # Constants for vectorbt enums (integers)
    SIZE_TYPE_PERCENT = 2  # Percent
    SIZE_TYPE_TARGET_PERCENT = 5  # TargetPercent
    DIRECTION_LONGONLY = 0  # LongOnly
    DIRECTION_BOTH = 2  # Both

    # ENTRY: MACD cross up + price above SMA
    if not in_position:
        # Only attempt entry if we have non-nan values for the necessary indicators
        if (
            macd_cross_up
            and (not np.isnan(close_i))
            and (not np.isnan(sma_i))
            and close_i > sma_i
        ):
            # initialize tracking state
            state["in_position"] = True
            state["entry_idx"] = i
            state["highest_price"] = high_i if not np.isnan(high_i) else close_i

            # Buy with 100% of portfolio (Percent = 1.0)
            return (1.0, SIZE_TYPE_PERCENT, DIRECTION_LONGONLY)

        # No order
        return (float("nan"), 0, 0)

    # POSITION IS OPEN: update highest price seen since entry
    if not np.isnan(high_i):
        # When starting a position, ensure highest_price is at least the entry high
        if state["highest_price"] is None:
            state["highest_price"] = high_i
        else:
            state["highest_price"] = max(state["highest_price"], high_i)

    # Compute trailing stop based on highest_since_entry and ATR
    trailing_stop = float("nan")
    if (
        state["highest_price"] is not None
        and np.isfinite(state["highest_price"])
        and not np.isnan(atr_i)
    ):
        trailing_stop = state["highest_price"] - float(trailing_mult) * float(atr_i)

    # EXIT conditions: MACD cross down OR close falls below trailing stop
    exit_by_macd = macd_cross_down
    exit_by_trail = (
        (not np.isnan(trailing_stop))
        and (not np.isnan(close_i))
        and (close_i < trailing_stop)
    )

    if exit_by_macd or exit_by_trail:
        # Reset state (we ask to target 0% size)
        state["in_position"] = False
        state["entry_idx"] = None
        state["highest_price"] = -np.inf

        # Use TargetPercent=0 to close the position fully
        return (0.0, SIZE_TYPE_TARGET_PERCENT, DIRECTION_BOTH)

    # Otherwise, no order
    return (float("nan"), 0, 0)
