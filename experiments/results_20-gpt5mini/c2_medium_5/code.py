# -*- coding: utf-8 -*-
"""
MACD + ATR trailing stop strategy utilities for vectorbt backtests.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- No numba is used.
- The order function returns tuples (size, size_type, direction) as plain Python values.
- SizeType and Direction enum integers are used directly to avoid importing vectorbt enums.
  According to vectorbt docs:
    SizeType.TargetPercent == 5
    Direction.LongOnly == 0

"""
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
    Compute MACD, signal line, ATR and SMA used by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] (index = datetime)
        macd_fast: MACD fast EMA window
        macd_slow: MACD slow EMA window
        macd_signal: MACD signal line window
        sma_period: Period for trend filter SMA
        atr_period: Period for ATR

    Returns:
        Dictionary with numpy arrays for keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'
    """
    # Basic checks
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["close", "high", "low"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv is missing required column: {col}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # MACD using vectorbt indicator
    macd_ind = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR using vectorbt indicator
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # SMA using pandas (simple moving average)
    # Use min_periods=1 to avoid all-NaN arrays; keep NaNs produced by indicators as-is
    sma_series = close.rolling(window=sma_period, min_periods=1).mean()

    # Extract numpy arrays (1D)
    macd_arr = np.asarray(macd_ind.macd) if hasattr(macd_ind, "macd") else np.asarray(macd_ind["macd"])
    signal_arr = np.asarray(macd_ind.signal) if hasattr(macd_ind, "signal") else np.asarray(macd_ind["signal"])
    atr_arr = np.asarray(atr_ind.atr) if hasattr(atr_ind, "atr") else np.asarray(atr_ind["atr"])
    sma_arr = np.asarray(sma_series)
    close_arr = np.asarray(close)
    high_arr = np.asarray(high)

    # Ensure all arrays are 1D and have the same length
    n = len(close_arr)
    for name, arr in ("macd", macd_arr), ("signal", signal_arr), ("atr", atr_arr), ("sma", sma_arr), ("high", high_arr):
        if arr.ndim != 1:
            arr = arr.ravel()
        if len(arr) != n:
            raise ValueError(f"Indicator '{name}' has length {len(arr)} but expected {n}")

    return {
        "macd": macd_arr.astype(float),
        "signal": signal_arr.astype(float),
        "atr": atr_arr.astype(float),
        "sma": sma_arr.astype(float),
        "close": close_arr.astype(float),
        "high": high_arr.astype(float),
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
    Order function for vectorbt's from_order_func (non-numba).

    Returns a tuple: (size, size_type, direction)
      - size: float (for TargetPercent this is target fraction 0..1)
      - size_type: int (use enum integer values to avoid importing enums)
      - direction: int (enum int)

    Strategy logic (long-only):
      - Entry when MACD crosses above signal AND price > 50-period SMA
      - Exit when MACD crosses below signal OR price < (highest_since_entry - trailing_mult * ATR)

    Notes:
      - Uses SizeType.TargetPercent == 5 to set target portfolio percent (1.0 == 100%)
      - Uses Direction.LongOnly == 0
      - If no action, returns (np.nan, 0, 2) so the wrapper treats it as no order.
    """
    # Defensive checks for context
    i = getattr(c, "i", None)
    if i is None:
        # Can't determine current index; do nothing
        return (np.nan, 0, 2)

    try:
        i = int(i)
    except Exception:
        return (np.nan, 0, 2)

    # Ensure arrays are numpy arrays
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 2)

    # Helper to safely access previous values
    def prev_val(arr: np.ndarray, idx: int):
        if idx <= 0:
            return np.nan
        return arr[idx - 1]

    # Current and previous MACD/Signal
    macd_now = macd[i]
    signal_now = signal[i]
    macd_prev = prev_val(macd, i)
    signal_prev = prev_val(signal, i)

    # Current price and indicators
    price_now = close[i]
    sma_now = sma[i]
    atr_now = atr[i]

    # Validate necessary numbers
    def is_finite(x):
        return np.isfinite(x)

    # Detect crossovers
    macd_cross_up = (
        is_finite(macd_prev)
        and is_finite(signal_prev)
        and is_finite(macd_now)
        and is_finite(signal_now)
        and (macd_prev <= signal_prev)
        and (macd_now > signal_now)
    )

    macd_cross_down = (
        is_finite(macd_prev)
        and is_finite(signal_prev)
        and is_finite(macd_now)
        and is_finite(signal_now)
        and (macd_prev >= signal_prev)
        and (macd_now < signal_now)
    )

    price_above_sma = is_finite(price_now) and is_finite(sma_now) and (price_now > sma_now)

    # Determine whether we are currently long (position_now > 0)
    # SignalContext exposes `position_now` according to vectorbt docs
    position_now = getattr(c, "position_now", None)
    if position_now is None:
        # Fallback: treat unknown as no position
        position_now = 0.0

    in_long = float(position_now) > 1e-12

    # Build entry mask array on the fly to find last entry index for trailing stop
    # Cross up mask (vectorized) - we compute full arrays here but that's OK for single-asset backtest
    macd_arr = macd
    signal_arr = signal
    close_arr = close
    sma_arr = sma

    # prev arrays
    macd_prev_arr = np.empty_like(macd_arr)
    signal_prev_arr = np.empty_like(signal_arr)
    macd_prev_arr[0] = np.nan
    signal_prev_arr[0] = np.nan
    macd_prev_arr[1:] = macd_arr[:-1]
    signal_prev_arr[1:] = signal_arr[:-1]

    valid_mask = np.isfinite(macd_prev_arr) & np.isfinite(signal_prev_arr) & np.isfinite(macd_arr) & np.isfinite(signal_arr)
    cross_up_mask = valid_mask & (macd_prev_arr <= signal_prev_arr) & (macd_arr > signal_arr)

    # Entry condition array: cross up AND price > sma
    entry_mask = cross_up_mask & np.isfinite(close_arr) & np.isfinite(sma_arr) & (close_arr > sma_arr)

    # Find the most recent entry index up to current bar
    entry_idxs = np.where(entry_mask[: i + 1])[0]
    last_entry_idx = int(entry_idxs[-1]) if entry_idxs.size > 0 else None

    # Determine trailing stop (only meaningful if in position and we have an entry idx)
    exit_on_trail = False
    if in_long and last_entry_idx is not None and is_finite(atr_now):
        # Highest high since entry (inclusive)
        if last_entry_idx <= i:
            highest_price = np.nanmax(high[last_entry_idx : i + 1])
            if np.isfinite(highest_price):
                trailing_level = highest_price - float(trailing_mult) * float(atr_now)
                if is_finite(price_now) and price_now < trailing_level:
                    exit_on_trail = True

    # Exit if MACD crosses down
    exit_on_macd = in_long and macd_cross_down

    # Entry signal
    entry_signal = (not in_long) and macd_cross_up and price_above_sma

    # Decision logic: priority to exit signals
    if exit_on_macd or exit_on_trail:
        # Close position: set target percent to 0
        # SizeType.TargetPercent == 5, Direction.LongOnly == 0
        return (0.0, 5, 0)

    if entry_signal:
        # Open full position: target 100% of portfolio
        return (1.0, 5, 0)

    # No action
    return (np.nan, 0, 2)
