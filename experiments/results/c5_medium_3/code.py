import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Tuple, Any
import importlib
import inspect
import re

# Small class to hold trading state across calls to order_func.
class _OrderState:
    def __init__(self):
        self.entry_index = None
        self.highest = None
        self.in_position = False

# Attempt to discover the Python-level Order namedtuple fields used by vectorbt by
# parsing the source of candidate modules.
def _discover_order_fields() -> Tuple[str, ...]:
    candidates = [
        'vectorbt.portfolio.nb',
        'vectorbt.portfolio',
        'vectorbt.portfolio.base',
        'vectorbt.portfolio.order',
        'vectorbt.portfolio.orders',
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        try:
            src = inspect.getsource(mod)
        except Exception:
            continue
        # Look for namedtuple('Order', [ ... ]) or namedtuple('Order', 'a b c') patterns
        m = re.search(r"namedtuple\('\s*Order\s*'\s*,\s*\[([^\]]+)\]\)", src)
        if m:
            # Extract comma-separated fields
            fields_raw = m.group(1)
            # Remove quotes and whitespace
            fields = [f.strip().strip("\"\'") for f in fields_raw.split(',') if f.strip()]
            if 'price' in fields and 'size' in fields:
                return tuple(fields)
        # Try alternative pattern: namedtuple('Order', 'a b c')
        m2 = re.search(r"namedtuple\('\s*Order\s*'\s*,\s*['\"]([\w\s]+)['\"]\)", src)
        if m2:
            fields = [f.strip() for f in m2.group(1).split() if f.strip()]
            if 'price' in fields and 'size' in fields:
                return tuple(fields)
    return tuple()

_ORDER_FIELD_NAMES: Tuple[str, ...] = _discover_order_fields()

# Helper to construct an order tuple matching the discovered order fields.
def _make_order_tuple(size: float, price: float, size_type: int, direction: int) -> tuple:
    if not _ORDER_FIELD_NAMES:
        # Fallback: 4-element tuple (size, price, size_type, direction)
        return (float(size), float(price), int(size_type), int(direction))

    vals: List[Any] = []
    for fname in _ORDER_FIELD_NAMES:
        if fname == 'size':
            vals.append(float(size))
        elif fname == 'price':
            vals.append(float(price))
        elif fname == 'size_type':
            vals.append(int(size_type))
        elif fname == 'direction':
            vals.append(int(direction))
        elif fname in ('fees', 'fee', 'commission'):
            vals.append(0.0)
        else:
            # Default heuristic: integer-like names -> int, else float
            if any(token in fname for token in ('type', 'direction', 'status', 'mode', 'side')):
                vals.append(0)
            else:
                vals.append(0.0)
    return tuple(vals)

# Module-level state instance. Will be (re)initialized at the first bar of each run.
_ORDER_STATE = _OrderState()


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> tuple:
    """
    Generate orders based on MACD cross entries with an ATR-based trailing stop.

    Return a tuple describing the order. We construct a tuple whose length and
    ordering matches vectorbt's internal Order namedtuple when possible to
    ensure the simulator accepts it.
    """
    global _ORDER_STATE

    i = int(c.i)
    pos = float(c.position_now) if hasattr(c, "position_now") else 0.0

    # Reset state at the beginning of each run
    if i == 0:
        _ORDER_STATE = _OrderState()

    # Helper: safe check for MACD cross up
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        a, b = macd, signal
        if np.isnan(a[idx]) or np.isnan(b[idx]) or np.isnan(a[idx - 1]) or np.isnan(b[idx - 1]):
            return False
        return (a[idx] > b[idx]) and (a[idx - 1] <= b[idx - 1])

    # Helper: safe check for MACD cross down
    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        a, b = macd, signal
        if np.isnan(a[idx]) or np.isnan(b[idx]) or np.isnan(a[idx - 1]) or np.isnan(b[idx - 1]):
            return False
        return (a[idx] < b[idx]) and (a[idx - 1] >= b[idx - 1])

    # Safe current values
    close_val = float(close[i]) if (i < len(close)) else np.nan
    high_val = float(high[i]) if (i < len(high)) else np.nan
    sma_val = float(sma[i]) if (i < len(sma)) else np.nan
    atr_val = float(atr[i]) if (i < len(atr)) else np.nan

    # No-op tuple
    no_op = _make_order_tuple(np.nan, np.nan, 0, 0)

    # If flat -> check entry
    if pos == 0.0:
        # Clear stale state
        _ORDER_STATE.in_position = False
        _ORDER_STATE.entry_index = None
        _ORDER_STATE.highest = None

        # Entry: MACD crosses up and price is above SMA
        if macd_cross_up(i) and (not np.isnan(sma_val)) and (not np.isnan(close_val)) and (close_val > sma_val):
            _ORDER_STATE.in_position = True
            _ORDER_STATE.entry_index = i
            _ORDER_STATE.highest = high_val if (not np.isnan(high_val)) else close_val
            # Return a market order sized by percent (size_type=2)
            return _make_order_tuple(0.95, np.inf, 2, 1)

        return no_op

    # If in a long position -> check exits
    else:
        # Initialize highest if missing
        if _ORDER_STATE.highest is None:
            _ORDER_STATE.highest = high_val if (not np.isnan(high_val)) else close_val

        # Update highest with current high
        if not np.isnan(high_val):
            _ORDER_STATE.highest = max(_ORDER_STATE.highest, high_val)

        highest = _ORDER_STATE.highest

        # Compute trailing stop if ATR available
        trailing_stop_price = None
        if (not np.isnan(atr_val)) and (highest is not None) and (not np.isnan(highest)):
            trailing_stop_price = highest - (trailing_mult * atr_val)

        # Exit on MACD bearish cross
        if macd_cross_down(i):
            _ORDER_STATE.in_position = False
            _ORDER_STATE.entry_index = None
            _ORDER_STATE.highest = None
            return _make_order_tuple(-np.inf, np.inf, 2, 1)

        # Exit on trailing stop breach
        if (trailing_stop_price is not None) and (not np.isnan(close_val)) and (close_val < trailing_stop_price):
            _ORDER_STATE.in_position = False
            _ORDER_STATE.entry_index = None
            _ORDER_STATE.highest = None
            return _make_order_tuple(-np.inf, np.inf, 2, 1)

        # Hold
        return no_op


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute required indicators using vectorbt indicator implementations.

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    # Fallback for optional low column
    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)
    low_sr = ohlcv["low"].astype(float) if ("low" in ohlcv.columns) else close_sr

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = atr_ind.atr.values

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = sma_ind.ma.values

    return {
        "close": close_sr.values,
        "high": high_sr.values,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }