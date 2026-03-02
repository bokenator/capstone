"""
MACD + ATR trailing stop strategy implementation for vectorbt backtester.

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14) -> dict[str, np.ndarray]
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple[float, int, int]

Notes:
- Uses vectorbt indicators: vbt.MACD.run, vbt.ATR.run, vbt.MA.run
- Long-only strategy: entries on MACD cross above + price > 50-SMA
- Exits on MACD cross below OR price < (highest_since_entry - trailing_mult * ATR)

This module is pure Python (no numba).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert a pandas Series/DataFrame/NumPy array to a 1D numpy array.
    Preserves NaNs and dtype as float where applicable.
    """
    if isinstance(x, pd.DataFrame):
        # If a DataFrame with a single column, take the first column; otherwise ravel
        if x.shape[1] == 1:
            arr = x.iloc[:, 0].to_numpy()
        else:
            arr = x.to_numpy().ravel()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    # Ensure 1D
    arr = arr.reshape(-1)
    return arr.astype(float, copy=False)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close'] (case-sensitive as used in sample data).
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line period for MACD.
        sma_period: Period for the trend filter SMA.
        atr_period: Period for ATR.

    Returns:
        Dict with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Basic column checks
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close_sr = ohlcv["close"]
    high_sr = ohlcv["high"]
    low_sr = ohlcv["low"]

    # MACD
    macd_res = vbt.MACD.run(
        close_sr,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR
    atr_res = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # SMA (use MA indicator)
    ma_res = vbt.MA.run(close_sr, window=sma_period)

    # Convert outputs to 1D numpy arrays
    macd_arr = _to_1d_array(macd_res.macd)
    signal_arr = _to_1d_array(macd_res.signal)
    atr_arr = _to_1d_array(atr_res.atr)
    sma_arr = _to_1d_array(ma_res.ma)

    close_arr = _to_1d_array(close_sr)
    high_arr = _to_1d_array(high_sr)

    # Sanity: ensure all arrays have same length
    n = len(close_arr)
    for name, arr in ("macd", macd_arr), ("signal", signal_arr), ("atr", atr_arr), ("sma", sma_arr), ("high", high_arr):
        if len(arr) != n:
            raise ValueError(f"Indicator '{name}' length {len(arr)} does not match close length {n}")

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
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """Order function for vectorbt.from_order_func (no numba).

    Parameters provided by the backtester (see run_backtest wrapper):
      - c: simulation context with attributes like `i` and `position_now`.
      - close, high, macd, signal, atr, sma: numpy arrays of indicators.
      - trailing_mult: multiplier for ATR used in trailing stop (e.g., 2.0).

    Return:
      Tuple (size, size_type, direction).
        - size: float (use np.nan to indicate no order)
        - size_type: one of SizeType enum values
        - direction: one of Direction enum values

    Strategy logic (long-only):
      Entry: MACD line crosses above Signal AND price > 50-SMA
      Exit: MACD line crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Note: The function determines "highest_since_entry" by locating the most recent
    deterministic entry signal (MACD cross above + price>sma) up to the current index.
    This mirrors the deterministic entry rule used by the strategy and is sufficient
    for a single-asset, long-only backtest.
    """
    # Defensive checks
    i = getattr(c, "i", None)
    if i is None:
        # Cannot determine index -> no action
        return (np.nan, SizeType.Amount, Direction.Both)

    if not (0 <= i < len(close)):
        return (np.nan, SizeType.Amount, Direction.Both)

    # Current position (in asset units). Use it to decide entries vs exits.
    pos_now = float(getattr(c, "position_now", 0.0))

    # At the first bar we cannot compute crosses -> no action
    if i == 0:
        return (np.nan, SizeType.Amount, Direction.Both)

    # If essential indicators are NaN at current or previous bar, skip
    if (
        np.isnan(macd[i])
        or np.isnan(signal[i])
        or np.isnan(macd[i - 1])
        or np.isnan(signal[i - 1])
        or np.isnan(sma[i])
        or np.isnan(close[i])
    ):
        return (np.nan, SizeType.Amount, Direction.Both)

    # MACD cross detection at current bar
    macd_prev = macd[i - 1]
    signal_prev = signal[i - 1]
    macd_curr = macd[i]
    signal_curr = signal[i]

    cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
    cross_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    entry_cond = cross_up and (close[i] > sma[i])

    # Locate the most recent deterministic entry index (based on indicators) up to i
    last_entry_idx = None
    # iterate backwards from i to 1 (entry requires i-1 to exist)
    for j in range(i, 0, -1):
        # ensure indicators exist at j and j-1
        if (
            np.isnan(macd[j])
            or np.isnan(signal[j])
            or np.isnan(macd[j - 1])
            or np.isnan(signal[j - 1])
            or np.isnan(sma[j])
            or np.isnan(close[j])
        ):
            continue

        if (macd[j - 1] <= signal[j - 1]) and (macd[j] > signal[j]) and (close[j] > sma[j]):
            last_entry_idx = j
            break

    # Compute trailing stop condition if we have an entry and are in a position
    exit_trail = False
    if (last_entry_idx is not None) and (pos_now > 0):
        # take highest high since last entry (inclusive)
        seg = high[last_entry_idx : i + 1]
        if seg.size > 0 and not np.all(np.isnan(seg)) and not np.isnan(atr[i]):
            highest = float(np.nanmax(seg))
            threshold = highest - float(trailing_mult) * float(atr[i])
            # protect against NaN threshold
            if not np.isnan(threshold):
                exit_trail = close[i] < threshold

    # Decision logic
    # Enter long: no current position and entry condition met
    if (pos_now <= 0) and entry_cond:
        # Target 100% allocation (TargetPercent = 1.0)
        return (1.0, int(SizeType.TargetPercent), int(Direction.LongOnly))

    # Exit long: currently long and either MACD cross down or trailing stop hit
    if (pos_now > 0) and (cross_down or exit_trail):
        # Set target percent to 0 -> exit fully
        return (0.0, int(SizeType.TargetPercent), int(Direction.LongOnly))

    # Otherwise, no action
    return (np.nan, SizeType.Amount, Direction.Both)
