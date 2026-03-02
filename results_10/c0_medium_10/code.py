"""
MACD + ATR Trailing Stop strategy for vectorbt

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
    -> dict of numpy arrays: macd, signal, atr, sma, close, high

- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)
    -> tuple (size, size_type, direction)

Notes:
- No numba usage.
- Returns simple Python tuples from order_func.

Assumptions for order tuple enum ints (compatible with vectorbt defaults):
- size_type: 0 -> Amount (absolute asset amount)
- direction: 1 -> Long (buy), 2 -> Short (sell)

The order function tries to use portfolio cash and current position size from the context object (common attributes used by vectorbt). Falls back to conservative defaults if attributes are missing.
"""
from __future__ import annotations

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
    """
    Compute MACD, Signal, ATR and SMA used by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open','high','low','close','volume'] (at minimum high/low/close required)
        macd_fast, macd_slow, macd_signal: MACD parameters
        sma_period: Period for trend filter SMA
        atr_period: Period for ATR

    Returns:
        dict with keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high' each mapping to 1D numpy arrays

    Notes:
        - Uses exponential moving averages (ewm with adjust=False) for MACD and Signal.
        - Uses Wilder-style smoothing for ATR (ewm with alpha=1/atr_period).
        - SMA uses min_periods=sma_period so that initial values are NaN until enough bars.

    """
    # Validate required columns
    required = ["high", "low", "close"]
    for col in required:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    # Ensure float dtype
    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # --- MACD ---
    # Exponential moving averages
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # --- ATR ---
    prev_close = close_s.shift(1)
    tr_1 = high_s - low_s
    tr_2 = (high_s - prev_close).abs()
    tr_3 = (low_s - prev_close).abs()
    tr = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    # Wilder's smoothing (exponential with alpha=1/period)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # --- SMA trend filter ---
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    return {
        "macd": macd.values.astype(float),
        "signal": signal.values.astype(float),
        "atr": atr.values.astype(float),
        "sma": sma.values.astype(float),
        "close": close_s.values.astype(float),
        "high": high_s.values.astype(float),
    }


def _is_cross_up(macd: np.ndarray, signal: np.ndarray, idx: int) -> bool:
    if idx <= 0:
        return False
    a0 = macd[idx - 1]
    b0 = signal[idx - 1]
    a1 = macd[idx]
    b1 = signal[idx]
    if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
        return False
    return (a0 <= b0) and (a1 > b1)


def _is_cross_down(macd: np.ndarray, signal: np.ndarray, idx: int) -> bool:
    if idx <= 0:
        return False
    a0 = macd[idx - 1]
    b0 = signal[idx - 1]
    a1 = macd[idx]
    b1 = signal[idx]
    if np.isnan(a0) or np.isnan(b0) or np.isnan(a1) or np.isnan(b1):
        return False
    return (a0 >= b0) and (a1 < b1)


def _safe_get_position_info(c: Any, high: np.ndarray, idx: int):
    """
    Try to extract position information from the context object c.

    Returns:
        (is_open: bool, pos_size: float|None, entry_idx: int|None, highest_price: float|None)
    """
    is_open = False
    pos_size = None
    entry_idx = None
    highest_price = None

    # Common attribute path
    pos = getattr(c, "position", None)
    if pos is None:
        # Some contexts might use 'pos' name
        pos = getattr(c, "pos", None)

    if pos is None:
        return is_open, pos_size, entry_idx, highest_price

    # is_open
    if hasattr(pos, "is_open"):
        try:
            is_open = bool(getattr(pos, "is_open"))
        except Exception:
            is_open = False
    else:
        # Fallback: check size/amount
        size_attr = None
        for name in ("size", "amount", "qty"):
            if hasattr(pos, name):
                size_attr = getattr(pos, name)
                break
        try:
            is_open = bool(size_attr) if size_attr is not None else False
        except Exception:
            is_open = False

    # pos_size
    for name in ("size", "amount", "qty"):
        if hasattr(pos, name):
            try:
                pos_size = float(getattr(pos, name))
                break
            except Exception:
                pos_size = None

    # highest_price or max_price
    for name in ("highest", "highest_price", "max_price", "high_price", "max_high"):
        if hasattr(pos, name):
            try:
                highest_price = float(getattr(pos, name))
                break
            except Exception:
                highest_price = None

    # entry index
    for name in ("entry_idx", "entry_i", "entry_index", "entry_row"):
        if hasattr(pos, name):
            try:
                entry_idx = getattr(pos, name)
                # Cast to int if possible
                if entry_idx is not None and not np.isnan(entry_idx):
                    entry_idx = int(entry_idx)
                else:
                    entry_idx = None
                break
            except Exception:
                entry_idx = None

    # If highest_price is not available but entry_idx is, compute highest from high array
    if highest_price is None and entry_idx is not None:
        try:
            if 0 <= entry_idx <= idx:
                highest_price = float(np.nanmax(high[int(entry_idx) : int(idx) + 1]))
        except Exception:
            highest_price = None

    return is_open, pos_size, entry_idx, highest_price


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
    Order function for vectorbt.from_order_func.

    Returns a tuple (size, size_type, direction).

    Behavior:
    - Entry: when MACD crosses above Signal AND price > SMA AND no open position -> enter long
    - Exit: when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR) -> close position

    Implementation details:
    - Uses context object c to inspect current index (c.i), cash (c.cash) and position info (c.position).
    - Entry size: attempts to invest all available cash (c.cash / price). Falls back to size 1.0.
    - Exit size: attempts to close full position by selling pos_size if available; falls back to size 1.0.

    IMPORTANT: This function returns plain Python tuple. The outer runner will convert it to a vectorbt Order.

    """
    i = int(getattr(c, "i", 0))

    # Protect index bounds
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    price = float(close[i])

    # Check indicator-based signals
    entry_signal = _is_cross_up(macd, signal, i)
    exit_signal = _is_cross_down(macd, signal, i)

    # Trend filter: price must be above SMA
    price_above_sma = False
    try:
        sma_val = float(sma[i])
        if not np.isnan(sma_val):
            price_above_sma = price > sma_val
    except Exception:
        price_above_sma = False

    # Gather position info
    is_open, pos_size, entry_idx, highest_price = _safe_get_position_info(c, high, i)

    # Trailing stop check (only if we have ATR and a recorded highest_price)
    trailing_hit = False
    try:
        atr_val = float(atr[i])
        if not np.isnan(atr_val) and highest_price is not None and not np.isnan(highest_price):
            thresh = highest_price - trailing_mult * atr_val
            trailing_hit = price < thresh
    except Exception:
        trailing_hit = False

    # --- Exits ---
    if is_open:
        if exit_signal or trailing_hit:
            # Determine size to sell: prefer pos_size from context
            sell_size = None
            if pos_size is not None and not np.isnan(pos_size) and pos_size > 0:
                sell_size = float(pos_size)
            else:
                # Fallback: try to use available cash to estimate one unit (will simply close minimum)
                sell_size = 1.0

            # size_type=0 -> Amount (absolute asset units), direction=2 -> Short/Sell
            return (sell_size, 0, 2)

    # --- Entries ---
    else:
        if entry_signal and price_above_sma:
            # Compute size to invest (try to use full available cash)
            size = None
            # c.cash is common attribute for available cash
            cash = None
            for attr in ("cash", "portfolio_cash", "available_cash"):
                if hasattr(c, attr):
                    try:
                        cash = float(getattr(c, attr))
                        break
                    except Exception:
                        cash = None

            if cash is not None and not np.isnan(cash) and price > 0:
                # Invest all cash (allow fractional shares)
                size = float(cash / price)
                # Avoid extremely small sizes
                if size <= 0:
                    size = 1.0
            else:
                size = 1.0

            # size_type=0 -> Amount (absolute asset units), direction=1 -> Long/Buy
            return (size, 0, 1)

    # No action
    return (np.nan, 0, 0)
