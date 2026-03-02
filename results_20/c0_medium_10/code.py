# Generated trading strategy combining MACD crossover entries with ATR-based trailing stops.
# compute_indicators and order_func are exported for the backtest runner.

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute MACD, Signal, ATR and SMA from OHLCV data.

    Args:
        ohlcv: DataFrame with columns at least ['open', 'high', 'low', 'close']
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal EMA period
        sma_period: Period for trend SMA
        atr_period: Period for ATR

    Returns:
        Dictionary with numpy arrays for keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'

    Notes:
        - Uses pandas ewm for EMAs (adjust=False)
        - ATR uses Wilder smoothing approximation via ewm with alpha=1/atr_period
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise KeyError(f"ohlcv must contain columns: {required_cols}")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD
    # Use EMA with adjust=False
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # ATR (True Range -> Wilder's smoothed ATR using ewm with alpha=1/atr_period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing approximation
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    return {
        "macd": macd.values.astype(float),
        "signal": signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
        "close": close.values.astype(float),
        "high": high.values.astype(float),
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
    """Order function implementing MACD crossover entries with ATR trailing stops.

    Returns a tuple accepted by the backtest wrapper: (size, size_type, direction)
    - size: float (use np.nan for no action)
    - size_type: int (enum value; we use 1 for percent allocation)
    - direction: int (1 for long entries, 2 for short/exits)

    Notes:
        - This function is stateful: it tracks whether a position is open and the highest
          price since entry using attributes on the function object.
        - MUST NOT use numba. Pure Python implementation.
    """
    # Initialize persistent state on the function
    if not hasattr(order_func, "_state"):
        order_func._state = {
            "in_position": False,
            "entry_idx": -1,
            "entry_high": -np.inf,
        }

    state = order_func._state

    # Safely extract index
    if not hasattr(c, "i"):
        # If context does not provide index, do nothing
        return (np.nan, 0, 0)

    i = int(c.i)

    # Helper to safely get array value
    def _val(arr: np.ndarray, idx: int) -> float:
        if idx < 0 or idx >= arr.shape[0]:
            return float("nan")
        v = arr[idx]
        try:
            return float(v)
        except Exception:
            return float("nan")

    price = _val(close, i)
    high_i = _val(high, i)
    macd_i = _val(macd, i)
    signal_i = _val(signal, i)
    atr_i = _val(atr, i)
    sma_i = _val(sma, i)

    # Detect MACD cross up / down (use previous bar)
    cross_up = False
    cross_down = False
    if i > 0:
        prev_macd = _val(macd, i - 1)
        prev_signal = _val(signal, i - 1)
        if not np.isnan(prev_macd) and not np.isnan(prev_signal) and not np.isnan(macd_i) and not np.isnan(signal_i):
            cross_up = (macd_i > signal_i) and (prev_macd <= prev_signal)
            cross_down = (macd_i < signal_i) and (prev_macd >= prev_signal)

    # Long entry condition
    if not state["in_position"]:
        if cross_up and (not np.isnan(price)) and (not np.isnan(sma_i)) and (price > sma_i):
            # Enter long: allocate 100% (size=1.0, size_type=1 -> percent, direction=1 -> long)
            state["in_position"] = True
            state["entry_idx"] = i
            state["entry_high"] = high_i if not np.isnan(high_i) else price
            return (1.0, 1, 1)

        # No action
        return (np.nan, 0, 0)

    # If in position, check exits and update highest price since entry
    if state["in_position"]:
        if not np.isnan(high_i):
            # Update highest observed price since entry
            if high_i > state["entry_high"]:
                state["entry_high"] = high_i

        # Exit on MACD cross down
        if cross_down:
            state["in_position"] = False
            # Exit (direction=2 -> short/sell)
            return (1.0, 1, 2)

        # Exit on ATR-based trailing stop
        if not np.isnan(atr_i) and not np.isneginf(state["entry_high"]) and not np.isnan(state["entry_high"]):
            trailing_stop = state["entry_high"] - float(trailing_mult) * atr_i
            if (not np.isnan(price)) and (price < trailing_stop):
                state["in_position"] = False
                return (1.0, 1, 2)

        # Otherwise, hold (no new order)
        return (np.nan, 0, 0)
