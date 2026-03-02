# Generated strategy: MACD crossover entries with ATR-based trailing stops
# Exports:
# - compute_indicators
# - order_func

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state to track open position and highest price since entry.
# NOTE: This is a simple Python-side stateful approach and assumes the
# backtest runs sequentially in a single process (which is true for
# vectorbt when use_numba=False). Not thread-safe.
_STATE: Dict[str, Any] = {
    "in_position": False,
    "entry_idx": None,
    "highest_price": -np.inf,
}


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

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close',...].
        macd_fast: Fast EMA window for MACD.
        macd_slow: Slow EMA window for MACD.
        macd_signal: Signal line EMA window for MACD.
        sma_period: Period for trend filter SMA.
        atr_period: Period for ATR.

    Returns:
        Dict with numpy arrays: 'macd', 'signal', 'atr', 'sma', 'close', 'high'

    Notes:
        - Uses vectorbt indicators for MACD and ATR to ensure consistent
          handling with the backtester.
        - Outputs are 1D numpy arrays aligned with ohlcv.index.
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ("high", "low", "close"):
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv is missing required column: {col}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD (vectorbt: fast_window, slow_window, signal_window)
    macd_ind = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # ATR (vectorbt: window)
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # SMA using pandas (min_periods = sma_period to honor warmup)
    sma_series = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # Extract numpy arrays (1D)
    macd_arr = np.asarray(macd_ind.macd).astype(float)
    signal_arr = np.asarray(macd_ind.signal).astype(float)
    atr_arr = np.asarray(atr_ind.atr).astype(float)
    sma_arr = np.asarray(sma_series).astype(float)
    close_arr = np.asarray(close).astype(float)
    high_arr = np.asarray(high).astype(float)

    # Sanity: ensure all arrays have same length
    n = len(close_arr)
    for name, arr in ("macd", macd_arr), ("signal", signal_arr), ("atr", atr_arr), ("sma", sma_arr), ("high", high_arr):
        if len(arr) != n:
            raise ValueError(f"Indicator '{name}' length {len(arr)} does not match close length {n}")

    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": close_arr,
        "high": high_arr,
    }


def _is_valid_number(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float = 2.0,
) -> Any:
    """
    Order function implementing the strategy logic.

    Args:
        c: vectorbt order context (provides .i and .position_now, etc.).
        close: 1D numpy array of close prices.
        high: 1D numpy array of high prices.
        macd: 1D numpy array of MACD line.
        signal: 1D numpy array of MACD signal line.
        atr: 1D numpy array of ATR values.
        sma: 1D numpy array of SMA values.
        trailing_mult: Multiplier for ATR when computing trailing stop.

    Returns:
        Tuple (size, size_type, direction) or np.nan-tuple for no-op.

    Strategy rules (long-only):
        - Entry: MACD crosses above Signal AND price > SMA(50)
        - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
    """
    global _STATE

    # Defensive checks
    if not hasattr(c, "i"):
        # If context doesn't provide index, don't place orders
        return (np.nan, 0, 0)

    idx = int(c.i)

    # Bounds check
    if idx < 0 or idx >= len(close):
        return (np.nan, 0, 0)

    # Read current values (may be NaN)
    price = float(close[idx]) if _is_valid_number(close[idx]) else np.nan
    high_price = float(high[idx]) if _is_valid_number(high[idx]) else np.nan
    curr_macd = float(macd[idx]) if _is_valid_number(macd[idx]) else np.nan
    curr_signal = float(signal[idx]) if _is_valid_number(signal[idx]) else np.nan
    curr_sma = float(sma[idx]) if _is_valid_number(sma[idx]) else np.nan
    curr_atr = float(atr[idx]) if _is_valid_number(atr[idx]) else np.nan

    # Update internal state based on actual portfolio position reported by context
    position_now = float(getattr(c, "position_now", 0.0))

    if position_now > 0:
        if not _STATE["in_position"]:
            # New position detected
            _STATE["in_position"] = True
            _STATE["entry_idx"] = idx
            _STATE["highest_price"] = high_price if _is_valid_number(high_price) else -np.inf
        else:
            # Update highest price since entry
            if _is_valid_number(high_price):
                _STATE["highest_price"] = max(_STATE["highest_price"], high_price)
    else:
        # No open position
        if _STATE["in_position"]:
            # Reset state on position close
            _STATE["in_position"] = False
            _STATE["entry_idx"] = None
            _STATE["highest_price"] = -np.inf

    # Determine crossovers (use previous bar)
    prev_macd = macd[idx - 1] if idx > 0 else np.nan
    prev_signal = signal[idx - 1] if idx > 0 else np.nan

    macd_cross_above = (
        _is_valid_number(prev_macd)
        and _is_valid_number(prev_signal)
        and _is_valid_number(curr_macd)
        and _is_valid_number(curr_signal)
        and (prev_macd <= prev_signal)
        and (curr_macd > curr_signal)
    )

    macd_cross_below = (
        _is_valid_number(prev_macd)
        and _is_valid_number(prev_signal)
        and _is_valid_number(curr_macd)
        and _is_valid_number(curr_signal)
        and (prev_macd >= prev_signal)
        and (curr_macd < curr_signal)
    )

    # Entry condition: MACD crosses above signal AND price above SMA
    entry_cond = (not _STATE["in_position"]) and macd_cross_above and _is_valid_number(price) and _is_valid_number(curr_sma) and (price > curr_sma)

    # Trailing stop condition: price falls below (highest_since_entry - trailing_mult * ATR)
    trailing_cond = False
    if _STATE["in_position"] and _STATE["highest_price"] != -np.inf and _is_valid_number(curr_atr) and _is_valid_number(price):
        stop_price = _STATE["highest_price"] - (trailing_mult * curr_atr)
        # Only trigger if stop_price is finite
        if _is_valid_number(stop_price):
            trailing_cond = price < stop_price

    # Exit condition: either MACD crosses below OR trailing stop
    exit_cond = _STATE["in_position"] and (macd_cross_below or trailing_cond)

    # Decide orders
    # For robustness, return tuples that the wrapper converts to orders.
    # We use simple sizing:
    # - Entry: buy 1 unit (Amount)
    # - Exit: set target amount to 0 (TargetAmount)
    # SizeType mapping (by index): 0=Amount,1=Value,2=Percent,3=TargetAmount,4=TargetValue,5=TargetPercent
    # Direction mapping (by index): 0=LongOnly,1=ShortOnly,2=Both
    if entry_cond:
        # Buy 1 unit
        return (1.0, 0, 0)

    if exit_cond:
        # Target amount = 0 -> close position
        return (0.0, 3, 0)

    # No action
    return (np.nan, 0, 0)
