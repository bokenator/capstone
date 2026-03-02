"""
MACD + ATR trailing stop strategy implementation for vectorbt backtests.
Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

This module intentionally avoids Numba and vbt.portfolio.nb.* usage.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert various array-like objects to a 1D numpy float array."""
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    arr = np.asarray(arr, dtype=float)

    # If DataFrame with single column, flatten
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]

    # Ensure 1D
    if arr.ndim != 1:
        arr = arr.ravel()

    return arr


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute MACD (macd, signal), ATR, and SMA from OHLCV dataframe.

    Returns a dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'.

    All outputs are 1D numpy arrays of the same length as the input.
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame with at least 'high','low','close' columns")

    for col in ("high", "low", "close"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # MACD
    macd_res = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = _to_1d_array(macd_res.macd)
    signal_arr = _to_1d_array(macd_res.signal)

    # ATR
    atr_res = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    # ATR.run outputs `tr` and `atr` - we want `atr`
    atr_arr = _to_1d_array(atr_res.atr)

    # SMA - use min_periods=sma_period so initial values are NaN until warmup
    sma_series = close_s.rolling(window=sma_period, min_periods=sma_period).mean()
    sma_arr = _to_1d_array(sma_series)

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": _to_1d_array(close_s),
        "high": _to_1d_array(high_s),
    }


def _make_order_func() -> Callable:
    """
    Create an order function closure that tracks the highest price since entry
    and applies a trailing stop based on ATR.

    The returned function signature matches what vbt.Portfolio.from_order_func
    will call: order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

    Returns:
        A callable order function.
    """

    # Per-instance state (kept inside closure). We avoid module-level globals.
    state: Dict[str, Any] = {
        "in_position": False,  # whether we believe we are in a position
        "highest": np.nan,     # highest price seen since entry
        "entry_idx": None,
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
        Order function to be used with vectorbt.Portfolio.from_order_func.

        Returns a tuple (size, size_type, direction):
          - size: float number of units (or np.nan for no order)
          - size_type: int (0 = Amount)
          - direction: int (0 = LongOnly/buy, 1 = ShortOnly, 2 = Both)

        Notes:
          - Long-only entries are created when MACD crosses above Signal and
            price > SMA.
          - Exits occur when MACD crosses below Signal or when price falls
            below (highest_since_entry - trailing_mult * ATR).
        """
        # Ensure numpy arrays
        close = _to_1d_array(close)
        high = _to_1d_array(high)
        macd = _to_1d_array(macd)
        signal = _to_1d_array(signal)
        atr = _to_1d_array(atr)
        sma = _to_1d_array(sma)

        # Current index
        i = int(c.i) if hasattr(c, "i") else 0

        # Reset state at the start of a simulation (i == 0)
        if i == 0:
            state["in_position"] = False
            state["highest"] = np.nan
            state["entry_idx"] = None

        # Safety bounds check
        if i < 0 or i >= len(close):
            return (np.nan, 0, 2)

        # Read current values safely
        close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
        high_i = float(high[i]) if not np.isnan(high[i]) else np.nan
        atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan

        # MACD crossover checks (require previous bar to exist and be non-NaN)
        def macd_cross_up(idx: int) -> bool:
            if idx == 0:
                return False
            if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
                return False
            return (macd[idx] > signal[idx]) and (macd[idx - 1] <= signal[idx - 1])

        def macd_cross_down(idx: int) -> bool:
            if idx == 0:
                return False
            if np.isnan(macd[idx]) or np.isnan(signal[idx]) or np.isnan(macd[idx - 1]) or np.isnan(signal[idx - 1]):
                return False
            return (macd[idx] < signal[idx]) and (macd[idx - 1] >= signal[idx - 1])

        # Trend filter check
        price_above_sma = False
        if not np.isnan(close_i) and i < len(sma) and not np.isnan(sma[i]):
            price_above_sma = close_i > float(sma[i])

        # Current position size (before we're about to create an order)
        position_now = float(c.position_now) if hasattr(c, "position_now") else 0.0

        # ENTRY: MACD bullish crossover + price above SMA
        if position_now == 0.0 and macd_cross_up(i) and price_above_sma:
            # Initialize trailing-high at the price of entry bar (use high if available)
            state["in_position"] = True
            state["highest"] = high_i if not np.isnan(high_i) else close_i
            state["entry_idx"] = i

            # Buy 1 unit (Amount-based)
            return (1.0, 0, 0)

        # If we have a position open, manage trailing stop and MACD exit
        if position_now > 0.0:
            # If state wasn't initialized (e.g., resumed a live position), initialize using current bar
            if not state["in_position"]:
                state["in_position"] = True
                state["highest"] = high_i if not np.isnan(high_i) else close_i
                state["entry_idx"] = i

            # Update highest since entry
            if not np.isnan(high_i):
                if np.isnan(state["highest"]) or high_i > state["highest"]:
                    state["highest"] = high_i

            # Trailing stop check
            if not np.isnan(state["highest"]) and not np.isnan(atr_i):
                threshold = state["highest"] - float(trailing_mult) * atr_i
                if not np.isnan(close_i) and close_i < threshold:
                    # Exit full position by selling current amount
                    size = float(position_now)
                    # Reset state
                    state["in_position"] = False
                    state["highest"] = np.nan
                    state["entry_idx"] = None
                    # Use direction=2 (Both) to allow closing the long
                    return (size, 0, 2)

            # MACD bearish crossover exit
            if macd_cross_down(i):
                size = float(position_now)
                state["in_position"] = False
                state["highest"] = np.nan
                state["entry_idx"] = None
                return (size, 0, 2)

        # Default: no order
        return (np.nan, 0, 2)

    return order_func


# Export a single order_func instance (closure) to be used by the backtest runner.
order_func = _make_order_func()
