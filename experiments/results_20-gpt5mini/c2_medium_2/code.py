"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- Long-only, single-asset strategy
- Uses module-level state to track open positions and highest price since entry
- No Numba is used
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


# Module-level state to track position info per column (supports single-asset by using col=0)
_POSITION_STATE: Dict[int, Dict[str, Any]] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict:
    """Compute indicators required by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at least 'high','low','close').
        macd_fast: MACD fast window.
        macd_slow: MACD slow window.
        macd_signal: MACD signal window.
        sma_period: Period for trend filter (SMA).
        atr_period: Period for ATR.

    Returns:
        Dict with NumPy arrays: 'close', 'high', 'macd', 'signal', 'atr', 'sma'

    Raises:
        ValueError: if required columns are missing.
    """
    # Validate input
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)
    low_sr = ohlcv["low"].astype(float)

    # MACD
    macd_res = vbt.MACD.run(
        close_sr,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )
    # ATR
    atr_res = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    # SMA (moving average)
    sma_res = vbt.MA.run(close_sr, window=sma_period)

    # Extract as 1D numpy arrays
    macd_line = np.asarray(macd_res.macd).ravel()
    signal_line = np.asarray(macd_res.signal).ravel()
    atr_arr = np.asarray(atr_res.atr).ravel()
    sma_arr = np.asarray(sma_res.ma).ravel()

    close_arr = np.asarray(close_sr).ravel()
    high_arr = np.asarray(high_sr).ravel()

    # Ensure all arrays have the same length
    n = len(close_arr)
    for name, arr in ("macd", macd_line), ("signal", signal_line), ("atr", atr_arr), ("sma", sma_arr), ("high", high_arr):
        if len(arr) != n:
            raise ValueError(f"Indicator '{name}' has unexpected length: {len(arr)} != {n}")

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_line,
        "signal": signal_line,
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
) -> Tuple[float, Any, Any]:
    """Order function used by vectorbt.Portfolio.from_order_func (use_numba=False).

    Parameters passed by the backtester (in this order):
    - close, high, macd, signal, atr, sma, trailing_mult

    Logic:
    - Entry: MACD line crosses above Signal AND price > SMA
    - Exit: MACD line crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns:
    - Tuple (size, size_type, direction)
      - size = np.nan -> no action
      - For entries: (1.0, SizeType.TargetPercent, Direction.LongOnly) -> target 100% allocation
      - For exits: (0.0, SizeType.TargetPercent, Direction.LongOnly) -> target 0% allocation
    """
    # Get current position in time
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0

    # Reset state at the start of a simulation (i == 0) to avoid cross-run contamination
    if i == 0:
        _POSITION_STATE.clear()

    # Initialize state for this column
    if col not in _POSITION_STATE:
        _POSITION_STATE[col] = {
            "in_position": False,
            "entry_idx": None,
            "highest": -np.inf,
        }

    state = _POSITION_STATE[col]

    # Safe access helper
    def safe_get(arr: np.ndarray, idx: int) -> float:
        if idx is None or idx < 0 or idx >= arr.shape[0]:
            return np.nan
        return float(arr[idx])

    close_i = safe_get(close, i)
    high_i = safe_get(high, i)
    atr_i = safe_get(atr, i)
    sma_i = safe_get(sma, i)

    # MACD & signal current and previous
    macd_i = safe_get(macd, i)
    signal_i = safe_get(signal, i)
    macd_prev = safe_get(macd, i - 1) if i - 1 >= 0 else np.nan
    signal_prev = safe_get(signal, i - 1) if i - 1 >= 0 else np.nan

    # Detect crosses (robust to NaNs)
    macd_cross_up = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_i)
        and not np.isnan(signal_i)
        and (macd_prev <= signal_prev)
        and (macd_i > signal_i)
    )

    macd_cross_down = (
        not np.isnan(macd_prev)
        and not np.isnan(signal_prev)
        and not np.isnan(macd_i)
        and not np.isnan(signal_i)
        and (macd_prev >= signal_prev)
        and (macd_i < signal_i)
    )

    price_above_sma = (not np.isnan(sma_i)) and (not np.isnan(close_i)) and (close_i > sma_i)

    # If currently not in position, check for entry
    if not state["in_position"]:
        if macd_cross_up and price_above_sma:
            # Place a long entry: target 100% allocation
            state["in_position"] = True
            state["entry_idx"] = i
            # Initialize highest as current high (fall back to close if NaN)
            state["highest"] = high_i if not np.isnan(high_i) else close_i
            return (1.0, SizeType.TargetPercent, Direction.LongOnly)

        # No action
        return (np.nan, 0, 0)

    # If in position, update highest_since_entry
    if state["in_position"]:
        if not np.isnan(high_i):
            # Only update highest if we've moved past entry
            if state["entry_idx"] is None or i >= state["entry_idx"]:
                if high_i > state["highest"]:
                    state["highest"] = high_i

        # Compute trailing stop (only if atr is available and we have a valid highest)
        trailing_stop_price = np.nan
        if not np.isnan(state["highest"]) and (state["highest"] != -np.inf) and not np.isnan(atr_i):
            trailing_stop_price = state["highest"] - float(trailing_mult) * atr_i

        # Check exit conditions
        exit_by_macd = macd_cross_down
        exit_by_trail = (
            (not np.isnan(trailing_stop_price))
            and (not np.isnan(close_i))
            and (close_i < trailing_stop_price)
        )

        if exit_by_macd or exit_by_trail:
            # Close position: target 0% allocation
            # Reset state after issuing exit
            state["in_position"] = False
            state["entry_idx"] = None
            state["highest"] = -np.inf
            return (0.0, SizeType.TargetPercent, Direction.LongOnly)

        # No action while holding
        return (np.nan, 0, 0)
