"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators
- order_func

Notes:
- Uses vectorbt indicators (MACD, ATR, MA)
- Trailing stop implemented using a small module-level state to track the highest price since an entry
- Long-only, single-asset. Entries use TargetPercent=1.0 (100% of capital). Exits set TargetPercent=0.0

CRITICAL: No numba usage.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state to track entry/highest price across calls. This is simple stateful
# approach suitable for single-asset, single-run usage in the provided backtest runner.
_STATE: Dict[str, Any] = {
    "pending_entry": False,  # we issued an entry order but position not confirmed yet
    "in_trade": False,       # we believe we have a live trade
    "entry_idx": None,       # index where entry was issued/confirmed
    "highest": -np.inf,      # highest high since entry
}

# Constants for returning tuple orders (we avoid importing vbt.portfolio.enums per instructions)
# These integer values are derived from vectorbt enums (SizeType.TargetPercent = 5, Direction/Both = 2)
_SIZE_TYPE_TARGET_PERCENT = 5
_DIRECTION_BOTH = 2


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute MACD, ATR and SMA indicators using vectorbt.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...]
        macd_fast, macd_slow, macd_signal: MACD params
        sma_period: period for trend filter SMA
        atr_period: ATR period

    Returns:
        Dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'
        Each value is a 1-D numpy.ndarray aligned to ohlcv.index.
    """
    # Basic validation
    required = ["high", "low", "close"]
    for col in required:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # MACD
    macd_res = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )
    macd_arr = np.asarray(macd_res.macd)
    signal_arr = np.asarray(macd_res.signal)

    # ATR
    atr_res = vbt.ATR.run(high, low, close, window=atr_period)
    atr_arr = np.asarray(atr_res.atr)

    # SMA (trend filter)
    ma_res = vbt.MA.run(close, window=sma_period)
    sma_arr = np.asarray(ma_res.ma)

    # Return arrays
    return {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": np.asarray(close),
        "high": np.asarray(high),
    }


def _is_nan_any(*vals: Optional[float]) -> bool:
    """Helper: return True if any of the provided values is NaN."""
    for v in vals:
        # Use numpy's isnan but handle non-numeric gracefully
        try:
            if np.isnan(v):
                return True
        except Exception:
            # If it's not a scalar (e.g., array), treat as not-NaN here
            continue
    return False


def _crossed_above(a: np.ndarray, b: np.ndarray, i: int) -> bool:
    if i <= 0:
        return False
    if _is_nan_any(a[i], b[i], a[i - 1], b[i - 1]):
        return False
    return (a[i - 1] <= b[i - 1]) and (a[i] > b[i])


def _crossed_below(a: np.ndarray, b: np.ndarray, i: int) -> bool:
    if i <= 0:
        return False
    if _is_nan_any(a[i], b[i], a[i - 1], b[i - 1]):
        return False
    return (a[i - 1] >= b[i - 1]) and (a[i] < b[i])


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
):
    """
    Order function for Portfolio.from_order_func (use_numba=False).

    Logic:
    - Entry when MACD crosses above Signal AND close > SMA
      -> issue TargetPercent=1.0 order (full allocation)
    - Exit when MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)
      -> issue TargetPercent=0.0 order (close position)

    Returns either:
    - tuple (size, size_type, direction) where size=np.nan means no action

    Notes:
    - Uses simple module-level state to track highest high since entry. The state
      gets reset when c.i == 0 (start of simulation).
    - Avoids numba usage.
    """
    global _STATE

    # Defensive conversions
    i = int(getattr(c, "i", 0))
    # Reset state at start of a run
    if i == 0:
        _STATE = {
            "pending_entry": False,
            "in_trade": False,
            "entry_idx": None,
            "highest": -np.inf,
        }

    # Safely get current values
    try:
        close_i = float(close[i])
    except Exception:
        close_i = float("nan")
    try:
        high_i = float(high[i])
    except Exception:
        high_i = float("nan")

    # If any of the primary indicators is NaN at this bar, do nothing
    if _is_nan_any(macd[i], signal[i], atr[i], sma[i], close_i, high_i):
        # No order -> return tuple with NaN size (wrapper will convert it)
        return (np.nan, 0, 0)

    # Helper flags
    macd_cross_above = _crossed_above(macd, signal, i)
    macd_cross_below = _crossed_below(macd, signal, i)

    # If we previously issued an entry order but position not yet confirmed,
    # we keep 'pending_entry' until we detect position (c.position_now) turned > 0.
    # When we detect position is active we flip to in_trade.
    if not _STATE["in_trade"] and _STATE["pending_entry"]:
        # position confirmation is typically visible on the next bar; check c.position_now
        try:
            pos_now = float(getattr(c, "position_now", 0.0))
        except Exception:
            pos_now = 0.0
        if pos_now > 0:
            _STATE["in_trade"] = True
            _STATE["pending_entry"] = False
            # ensure highest includes the bar where entry was issued
            if _STATE["entry_idx"] is not None:
                try:
                    hslice = high[_STATE["entry_idx"] : i + 1]
                    if len(hslice) > 0:
                        _STATE["highest"] = float(np.nanmax(hslice))
                except Exception:
                    _STATE["highest"] = max(_STATE.get("highest", -np.inf), high_i)

    # If currently in trade, update highest and check exit conditions
    if _STATE["in_trade"]:
        # update highest
        if not np.isnan(high_i):
            _STATE["highest"] = max(_STATE.get("highest", -np.inf), high_i)

        # Compute trailing stop price
        trailing_stop_price = None
        if not np.isnan(atr[i]):
            trailing_stop_price = _STATE["highest"] - float(trailing_mult) * float(atr[i])

        # Exit condition 1: MACD crosses below
        if macd_cross_below:
            # Issue exit (set target percent to 0.0)
            return (0.0, _SIZE_TYPE_TARGET_PERCENT, _DIRECTION_BOTH)

        # Exit condition 2: Price falls below trailing stop
        if (trailing_stop_price is not None) and (not np.isnan(trailing_stop_price)):
            if close_i < trailing_stop_price:
                return (0.0, _SIZE_TYPE_TARGET_PERCENT, _DIRECTION_BOTH)

    # If not in trade, check entry
    if (not _STATE["in_trade"]) and (not _STATE["pending_entry"]):
        # Entry: MACD crossed above and price above SMA
        if macd_cross_above and (close_i > float(sma[i])):
            # Issue entry: set target percent to 100%
            # Mark pending entry so we can track highest after position is confirmed
            _STATE["pending_entry"] = True
            _STATE["entry_idx"] = i
            _STATE["highest"] = high_i if not np.isnan(high_i) else -np.inf
            return (1.0, _SIZE_TYPE_TARGET_PERCENT, _DIRECTION_BOTH)

    # Default: no action
    return (np.nan, 0, 0)
