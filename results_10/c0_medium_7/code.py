"""
MACD + ATR Trailing Stop strategy implementation for vectorbt backtests.

Exports:
- compute_indicators(ohlcv, macd_fast, macd_slow, macd_signal, sma_period, atr_period)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- Long-only, single-asset strategy.
- Uses module-level state to track active entry and highest price since entry (works with use_numba=False).
- Returns simple Python tuples: (size, size_type, direction). The wrapper in the runner will convert those to Orders.

CRITICAL: Does not use numba decorators or vbt.portfolio.nb.* functions.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Module-level state to track entry and highest price since entry.
# This simple dict persists across calls of order_func when use_numba=False.
_STATE: Dict[str, Any] = {
    "in_position": False,
    "entry_idx": None,
    "high_since_entry": np.nan,
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
    Compute indicators required by the strategy: MACD (macd, signal), SMA, ATR.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...]
        macd_fast: fast EMA period for MACD
        macd_slow: slow EMA period for MACD
        macd_signal: signal EMA period for MACD
        sma_period: period for the trend SMA filter
        atr_period: period for ATR calculation

    Returns:
        Dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high' (all numpy arrays)
    """
    # Validate input columns
    required_cols = ["close", "high", "low"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain column '{col}'")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # MACD: EMA fast - EMA slow, signal = EMA(macd)
    # Use ewm with adjust=False (common practice in trading)
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter (use min_periods=1 to avoid all NaNs)
    sma = close_s.rolling(window=sma_period, min_periods=1).mean()

    # ATR: True range and rolling average
    prev_close = close_s.shift(1)
    tr_1 = high_s - low_s
    tr_2 = (high_s - prev_close).abs()
    tr_3 = (low_s - prev_close).abs()
    tr = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    # Use Wilder-style EMA for ATR (common approach) -> ewm with alpha=1/atr_period
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=1).mean()

    # Convert to numpy arrays
    out = {
        "macd": macd_line.to_numpy(dtype=float),
        "signal": signal_line.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "sma": sma.to_numpy(dtype=float),
        "close": close_s.to_numpy(dtype=float),
        "high": high_s.to_numpy(dtype=float),
    }

    return out


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function compatible with vectorbt.Portfolio.from_order_func (use_numba=False).

    Strategy rules (long-only):
    - Entry: MACD crosses above Signal AND close > SMA
    - Exit: MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    This function returns a tuple (size, size_type, direction). We use target percent sizing
    for entries/exits (1.0 for full allocation, 0.0 to exit). We attempt to use the
    vectorbt enums if available; otherwise fall back to integer codes.

    Args:
        c: context object provided by vectorbt (has attribute .i for current index)
        close, high, macd, signal, atr, sma: numpy arrays of indicator values
        trailing_mult: multiplier for ATR to compute stop distance

    Returns:
        (size, size_type, direction) tuple; size can be np.nan to indicate no action
    """
    i = int(getattr(c, "i", 0))

    # Provide access to SizeType and Direction enums if available; fall back to ints
    try:
        # Import inside function to avoid top-level dependency in environments
        from vectorbt.portfolio.enums import SizeType, Direction

        SIZE_TYPE_TARGET = SizeType.TargetPercent
        # Use Both to allow closing positions; LongOnly could be used but Both is safe
        DIR_BOTH = Direction.Both
    except Exception:
        # Fallback integer codes (these are safe defaults in many vbt versions):
        SIZE_TYPE_TARGET = 1  # commonly TargetPercent / Percent
        DIR_BOTH = 0

    # Helper: return no-op
    if i < 0 or i >= len(close):
        return (np.nan, SIZE_TYPE_TARGET, DIR_BOTH)

    # Current values
    close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
    macd_i = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_i = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan

    # Detect MACD cross up / cross down with previous bar
    cross_up = False
    cross_down = False
    if i > 0:
        prev_macd = float(macd[i - 1]) if not np.isnan(macd[i - 1]) else np.nan
        prev_signal = float(signal[i - 1]) if not np.isnan(signal[i - 1]) else np.nan
        if not np.isnan(prev_macd) and not np.isnan(prev_signal) and not np.isnan(macd_i) and not np.isnan(signal_i):
            cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)
            cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

    # Update / read module-level position state
    in_pos = bool(_STATE.get("in_position", False))

    # Entry logic
    should_enter = False
    if cross_up and (not np.isnan(sma_i)) and (close_i > sma_i):
        # Only enter if not already in our tracked position
        if not in_pos:
            should_enter = True

    # If we decide to enter, record entry state and return order to set target percent to 1.0
    if should_enter:
        _STATE["in_position"] = True
        _STATE["entry_idx"] = i
        # initialize high_since_entry with current bar's high (if available)
        if not np.isnan(high_i):
            _STATE["high_since_entry"] = high_i
        else:
            _STATE["high_since_entry"] = close_i

        # Target 100% allocation
        return (1.0, SIZE_TYPE_TARGET, DIR_BOTH)

    # If currently in a tracked position, update highest and check trailing stop and MACD exit
    if in_pos:
        # Update highest since entry
        prev_high = _STATE.get("high_since_entry", np.nan)
        if np.isnan(prev_high):
            _STATE["high_since_entry"] = high_i
        else:
            if not np.isnan(high_i):
                _STATE["high_since_entry"] = max(prev_high, high_i)

        # Compute trailing stop
        high_since_entry = _STATE.get("high_since_entry", np.nan)
        stop_price = np.nan
        if (not np.isnan(high_since_entry)) and (not np.isnan(atr_i)):
            stop_price = float(high_since_entry - trailing_mult * atr_i)

        price_below_stop = False
        if (not np.isnan(stop_price)) and (not np.isnan(close_i)):
            price_below_stop = close_i < stop_price

        # If exit condition met (MACD cross down OR price below trailing stop), exit
        if cross_down or price_below_stop:
            # Reset state
            _STATE["in_position"] = False
            _STATE["entry_idx"] = None
            _STATE["high_since_entry"] = np.nan

            # Issue order to set target percent to 0 (fully exit)
            return (0.0, SIZE_TYPE_TARGET, DIR_BOTH)

        # Otherwise, no change (stay in position)
        return (np.nan, SIZE_TYPE_TARGET, DIR_BOTH)

    # Default: no order
    return (np.nan, SIZE_TYPE_TARGET, DIR_BOTH)
