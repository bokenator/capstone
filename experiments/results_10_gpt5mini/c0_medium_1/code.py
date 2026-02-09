"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators(ohlcv: pd.DataFrame, macd_fast: int=12, macd_slow: int=26, macd_signal: int=9,
                     sma_period: int=50, atr_period: int=14) -> dict[str, np.ndarray]

- order_func(c, close, high, macd, signal, atr, sma, trailing_mult) -> tuple(size, size_type, direction)

Notes:
- Does NOT use numba or vectorbt numba helpers.
- Returns simple Python tuples from order_func.
- Uses TargetPercent sizing (size_type=2) and Long-only direction (direction=1) by default,
  but falls back to Amount sizing when available cash can be read from the context.

Strategy logic as requested:
- Entry: MACD line crosses above Signal AND close > sma(50)
- Exit: MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

Trailing stop uses highest high since entry and ATR (Wilder smoothing via ewm with alpha=1/atr_period)

"""
from typing import Any, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

# Constants for sizing/direction. These are integer values that correspond to vectorbt's
# SizeType and Direction enums. We avoid importing vectorbt.enums here per instructions.
# Common mapping in vectorbt:
#   SizeType.Amount == 0
#   SizeType.Percent == 1
#   SizeType.TargetPercent == 2
#   Direction.Both == 0
#   Direction.LongOnly == 1
_SIZE_TYPE_AMOUNT: int = 0
_SIZE_TYPE_TARGET_PERCENT: int = 2
_DIRECTION_LONG_ONLY: int = 1


def _first_available_float_attr(obj: Any, names: Iterable[str]) -> Optional[float]:
    """Return the first attribute from obj among names that can be converted to a finite float.
    Returns None if none found.
    """
    for name in names:
        val = getattr(obj, name, None)
        if val is None:
            continue
        try:
            f = float(val)
        except Exception:
            continue
        if np.isnan(f):
            continue
        return f
    return None


def _first_available_int_attr(obj: Any, names: Iterable[str]) -> Optional[int]:
    """Return the first attribute from obj among names that can be converted to an int >= 0.
    Returns None if none found.
    """
    for name in names:
        val = getattr(obj, name, None)
        if val is None:
            continue
        try:
            i = int(val)
        except Exception:
            continue
        if i < 0:
            continue
        return i
    return None


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
        ohlcv: DataFrame with columns ['open','high','low','close', ...]
        macd_fast: fast EMA period for MACD
        macd_slow: slow EMA period for MACD
        macd_signal: signal EMA period for MACD
        sma_period: period for trend SMA
        atr_period: period for ATR (Wilder smoothing)

    Returns:
        Dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'
        Each value is a numpy.ndarray aligned with the input ohlcv index.

    Notes:
        - Uses pandas ewm for EMA and Wilder-style ATR (ewm with alpha=1/atr_period, adjust=False).
        - Handles case-insensitive column names for OHLCV.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Map columns case-insensitively
    col_map = {col.lower(): col for col in ohlcv.columns}
    required = ["close", "high", "low"]
    for rc in required:
        if rc not in col_map:
            raise KeyError(f"ohlcv missing required column: {rc}")

    close_col = col_map["close"]
    high_col = col_map["high"]
    low_col = col_map["low"]

    # Work on copies to avoid modifying input
    close = ohlcv[close_col].astype(float).copy()
    high = ohlcv[high_col].astype(float).copy()
    low = ohlcv[low_col].astype(float).copy()

    # MACD (fast EMA - slow EMA) and signal line (EMA of MACD)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA (simple moving average) for trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR (True Range + Wilder smoothing). Use ewm with alpha=1/atr_period to mimic Wilder's smoothing.
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    # Return numpy arrays
    return {
        "macd": macd.to_numpy(dtype=float),
        "signal": signal.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "sma": sma.to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        "high": high.to_numpy(dtype=float),
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
    Order function to be used with vectorbt.Portfolio.from_order_func(..., use_numba=False).

    Returns tuple (size, size_type, direction):
      - size: float (for TargetPercent: target exposure e.g. 1.0 == 100%)
      - size_type: int (uses TargetPercent == 2 or Amount == 0)
      - direction: int (LongOnly == 1)

    Logic (long-only):
      - Entry: MACD crosses above Signal AND close > sma(50)
      - Exit: MACD crosses below Signal OR close < (highest_since_entry - trailing_mult * ATR)

    Notes:
      - Uses context attributes when available to read current position and available cash.
      - Avoids global state; computes highest_since_entry from the highs array using the entry index
        available in the context (if present).
    """
    # Validate input types
    if not isinstance(close, np.ndarray):
        raise TypeError("close must be a numpy.ndarray")

    # Get current bar index
    try:
        i = int(getattr(c, "i"))
    except Exception:
        raise RuntimeError("Order function context 'c' must provide integer attribute 'i'")

    # Bounds check
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Safe reads
    curr_close = float(close[i]) if not np.isnan(close[i]) else np.nan
    curr_high = float(high[i]) if not np.isnan(high[i]) else np.nan
    macd_val = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_val = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    atr_val = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_val = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # Determine if currently in position using context attributes when possible
    pos_val = _first_available_float_attr(c, ["pos", "size", "position_size", "position", "position_amount"]) 
    in_position = (pos_val > 0.0) if (pos_val is not None) else False

    # Entry index from context (try many common attribute names used by vectorbt contexts)
    entry_idx = _first_available_int_attr(c, ["entry_i", "last_entry_i", "entry_index", "entered_at"])

    # Compute highest since entry from highs if entry_idx is available
    highest_since_entry = None
    if entry_idx is not None and 0 <= entry_idx <= i:
        try:
            # use numpy nanmax to handle NaNs in the high series
            highest_since_entry = float(np.nanmax(high[entry_idx : i + 1]))
        except Exception:
            highest_since_entry = None

    # Fallback: if we have no entry_idx but pos_val suggests we are in position, we can attempt
    # to compute highest since the last non-zero pos by scanning backwards (expensive but safe).
    if in_position and highest_since_entry is None:
        # Find approximate start of position by scanning backwards for the first index where pos is zero.
        # We try to read a vectorized 'pos' attribute from the context (if any) to speed this up.
        pos_series = getattr(c, "pos_series", None)  # not guaranteed to exist
        if isinstance(pos_series, np.ndarray) and pos_series.shape and pos_series.ndim == 1:
            # find last time pos was zero before i
            nz = np.where(pos_series[: i + 1] == 0)[0]
            start_idx = nz[-1] + 1 if nz.size > 0 else 0
            try:
                highest_since_entry = float(np.nanmax(high[start_idx : i + 1]))
            except Exception:
                highest_since_entry = None

    # Compute previous MACD/Signal for cross detection
    if i == 0:
        prev_macd = np.nan
        prev_signal = np.nan
    else:
        prev_macd = macd[i - 1]
        prev_signal = signal[i - 1]

    cross_up = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and not np.isnan(macd_val)
        and not np.isnan(signal_val)
        and (prev_macd <= prev_signal)
        and (macd_val > signal_val)
    )

    cross_down = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and not np.isnan(macd_val)
        and not np.isnan(signal_val)
        and (prev_macd >= prev_signal)
        and (macd_val < signal_val)
    )

    # Trailing stop condition
    trailing_hit = False
    if in_position and (highest_since_entry is not None) and (not np.isnan(atr_val)):
        threshold = highest_since_entry - float(trailing_mult) * atr_val
        if not np.isnan(curr_close) and curr_close < threshold:
            trailing_hit = True

    # ENTRY: MACD cross up and price above SMA (and not already in position)
    if (not in_position) and cross_up and (not np.isnan(curr_close)) and (not np.isnan(sma_val)) and (curr_close > sma_val):
        # Try to read available cash from context to place an Amount order (explicit shares)
        cash_val = _first_available_float_attr(c, ["cash", "available_cash", "portfolio_cash", "cash_value"])  # heuristics
        if cash_val is not None and (not np.isnan(curr_close)) and curr_close > 0:
            shares = np.floor(cash_val / curr_close)
            if shares >= 1:
                return (float(shares), _SIZE_TYPE_AMOUNT, _DIRECTION_LONG_ONLY)
        # Fallback: use TargetPercent to request full exposure (1.0 == 100%)
        return (1.0, _SIZE_TYPE_TARGET_PERCENT, _DIRECTION_LONG_ONLY)

    # EXIT: MACD cross down OR trailing stop
    if in_position and (cross_down or trailing_hit):
        # Use TargetPercent=0.0 to clear exposure (reliable independent of share counts)
        return (0.0, _SIZE_TYPE_TARGET_PERCENT, _DIRECTION_LONG_ONLY)

    # No action
    return (np.nan, 0, 0)
