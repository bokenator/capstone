# Strategy combining MACD crossover entries with ATR-based trailing stops
# compute_indicators and order_func implementations for vectorbt backtester

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

# Try to import order_nb to build proper order objects expected by vectorbt's simulation
_order_nb_available = False
_order_nb = None
try:
    import vectorbt as vbt
    _nb_mod = getattr(vbt.portfolio, 'nb', None)
    if _nb_mod is not None and hasattr(_nb_mod, 'order_nb'):
        _order_nb = getattr(_nb_mod, 'order_nb')
        _order_nb_available = True
except Exception:
    _order_nb_available = False


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

    Returns a dictionary with keys:
      - close: np.ndarray (close prices)
      - high: np.ndarray (high prices)
      - macd: np.ndarray (MACD line)
      - signal: np.ndarray (MACD signal line)
      - atr: np.ndarray (ATR values)
      - sma: np.ndarray (50-period simple moving average)

    Handles NaNs and warmup periods by leaving initial values as np.nan.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame with columns open/high/low/close/volume")

    # Ensure required columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv is missing required column: {col}")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)
    open_s = ohlcv["open"].astype(float)

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd, signal_period)
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).to_numpy()
    signal_line = pd.Series(macd_line).ewm(span=macd_signal, adjust=False).mean().to_numpy()

    # SMA for trend filter
    sma = close_s.rolling(window=sma_period, min_periods=1).mean().to_numpy()

    # ATR: True Range then Wilder's moving average (EMA with alpha=1/n)
    high_low = high_s - low_s
    high_close = (high_s - close_s.shift(1)).abs()
    low_close = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean().to_numpy()

    return {
        "close": close_s.to_numpy(),
        "high": high_s.to_numpy(),
        "macd": macd_line,
        "signal": signal_line,
        "atr": atr,
        "sma": sma,
    }


# Global state for tracking entry index and highest price since entry
# This is a single-asset strategy so global state is acceptable for the backtester
_ENTRY_STATE: Dict[str, Any] = {
    "in_position": False,
    "entry_idx": -1,
    "highest_since_entry": np.nan,
    "entry_units": np.nan,
}


def _is_cross_up(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
    if idx <= 0:
        return False
    a_prev = arr1[idx - 1]
    b_prev = arr2[idx - 1]
    a_curr = arr1[idx]
    b_curr = arr2[idx]
    if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
        return False
    return (a_prev <= b_prev) and (a_curr > b_curr)


def _is_cross_down(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
    if idx <= 0:
        return False
    a_prev = arr1[idx - 1]
    b_prev = arr2[idx - 1]
    a_curr = arr1[idx]
    b_curr = arr2[idx]
    if np.isnan(a_prev) or np.isnan(b_prev) or np.isnan(a_curr) or np.isnan(b_curr):
        return False
    return (a_prev >= b_prev) and (a_curr < b_curr)


def _extract_from_ctx(ctx: Any) -> Tuple[int, float, float]:
    """
    Try to extract (idx, cash, current_pos) from an OrderContext-like object.
    Return (idx, cash, current_pos) where missing values become None (or np.nan for pos).
    """
    idx = None
    cash = None
    pos = None

    # Candidate attribute names
    idx_attrs = ["i", "idx", "index", "index_i", "row_idx"]
    cash_attrs = ["cash", "free_cash", "available_cash"]
    pos_attrs = ["pos", "position", "positions", "current_pos"]

    for a in idx_attrs:
        if hasattr(ctx, a):
            try:
                maybe = getattr(ctx, a)
                idx = int(maybe)
                break
            except Exception:
                pass
    for a in cash_attrs:
        if hasattr(ctx, a):
            try:
                maybe = getattr(ctx, a)
                cash = float(maybe)
                break
            except Exception:
                pass
    for a in pos_attrs:
        if hasattr(ctx, a):
            try:
                maybe = getattr(ctx, a)
                # If array-like, try first element
                if isinstance(maybe, np.ndarray):
                    pos = float(maybe[0]) if maybe.size > 0 else None
                else:
                    pos = float(maybe)
                break
            except Exception:
                pass

    return idx, cash, pos


def order_func(*args: Any, **kwargs: Any) -> Any:
    """
    Order function compatible with vectorbt.Portfolio.from_order_func (use_numba=False).

    The function is intentionally flexible in its argument parsing: vectorbt calls it with
    a number of positional arguments. In the Python runner the first argument is often an
    OrderContext object which contains useful information (index, cash, positions). The
    additional arrays passed from run_backtest are expected to be the last 7 positional
    arguments: (close, high, macd, signal, atr, sma, trailing_mult).

    Returns a vectorbt nb Order object constructed via vbt.portfolio.nb.order_nb when
    available, otherwise returns a numpy recarray fallback (less optimal).
    """
    # Parse context and arrays
    idx = None
    cash = None
    current_pos = None

    # If first arg is an OrderContext-like, extract from it
    if len(args) >= 1 and not isinstance(args[0], (int, np.integer, float, np.floating)):
        ctx = args[0]
        try:
            extracted_idx, extracted_cash, extracted_pos = _extract_from_ctx(ctx)
            if extracted_idx is not None:
                idx = extracted_idx
            if extracted_cash is not None:
                cash = extracted_cash
            if extracted_pos is not None:
                current_pos = extracted_pos
        except Exception:
            # ignore and fallback to positional parsing
            pass

    # If idx or cash still None, try positional parsing used earlier
    if idx is None:
        if len(args) >= 1:
            try:
                idx = int(args[0])
            except Exception:
                pass
    if cash is None:
        if len(args) >= 2:
            try:
                cash = float(args[1])
            except Exception:
                cash = None

    if idx is None:
        raise ValueError("Could not determine current index in order_func")

    # The last 7 args are assumed to be the arrays we passed in from run_backtest
    if len(args) < 8:
        # Not enough arguments to retrieve the indicator arrays; fail gracefully
        raise ValueError("order_func expected indicator arrays as additional arguments")

    close_arr = np.asarray(args[-7])
    high_arr = np.asarray(args[-6])
    macd_arr = np.asarray(args[-5])
    signal_arr = np.asarray(args[-4])
    atr_arr = np.asarray(args[-3])
    sma_arr = np.asarray(args[-2])
    trailing_mult = float(args[-1])

    # Current prices
    price = float(close_arr[idx])
    high_price = float(high_arr[idx])

    size_val = 0.0
    price_val = np.nan
    stop_val = np.nan
    limit_val = np.nan

    # Determine signals
    enter_signal = _is_cross_up(macd_arr, signal_arr, idx) and (not np.isnan(sma_arr[idx]) and price > sma_arr[idx])
    exit_signal_macd = _is_cross_down(macd_arr, signal_arr, idx)

    # Update highest since entry when in position
    if _ENTRY_STATE["in_position"] and _ENTRY_STATE["entry_idx"] >= 0:
        # update highest
        if np.isnan(_ENTRY_STATE["highest_since_entry"]):
            _ENTRY_STATE["highest_since_entry"] = high_price
        else:
            _ENTRY_STATE["highest_since_entry"] = max(_ENTRY_STATE["highest_since_entry"], high_price)

    # Trailing stop check
    trailing_stop_price = np.nan
    if _ENTRY_STATE["in_position"] and not np.isnan(atr_arr[idx]):
        trailing_stop_price = _ENTRY_STATE["highest_since_entry"] - trailing_mult * atr_arr[idx]
        # only valid if highest_since_entry is finite
        if np.isnan(trailing_stop_price):
            trailing_stop_price = np.nan

    exit_signal_trail = False
    if _ENTRY_STATE["in_position"] and not np.isnan(trailing_stop_price):
        if price < trailing_stop_price:
            exit_signal_trail = True

    # Decide orders
    if (not _ENTRY_STATE["in_position"]) and enter_signal:
        # Enter: buy 1 unit to ensure trades are executed in the backtest environment
        order_size = 1.0
        size_val = order_size
        price_val = price
        # Set state - assume immediate fill
        _ENTRY_STATE["in_position"] = True
        _ENTRY_STATE["entry_idx"] = idx
        _ENTRY_STATE["highest_since_entry"] = high_price
        _ENTRY_STATE["entry_units"] = order_size
    elif _ENTRY_STATE["in_position"] and (exit_signal_macd or exit_signal_trail):
        # Exit: sell entire position (market order)
        sell_units = None
        if current_pos is not None and not np.isnan(current_pos):
            sell_units = float(current_pos)
        elif not np.isnan(_ENTRY_STATE.get("entry_units", np.nan)):
            sell_units = float(_ENTRY_STATE.get("entry_units", 0.0))

        if sell_units is None or sell_units == 0.0:
            sell_units = 1.0

        size_val = -sell_units
        price_val = price
        # Reset state - assume immediate fill
        _ENTRY_STATE["in_position"] = False
        _ENTRY_STATE["entry_idx"] = -1
        _ENTRY_STATE["highest_since_entry"] = np.nan
        _ENTRY_STATE["entry_units"] = np.nan

    # If no order, return a no-order representation
    if size_val == 0.0:
        if _order_nb_available and _order_nb is not None:
            return _order_nb(np.nan)
        else:
            order_dtype = np.dtype([
                ("size", np.float64),
                ("price", np.float64),
                ("stop_price", np.float64),
                ("limit_price", np.float64),
            ])
            order_arr = np.array([(np.nan, np.nan, np.nan, np.nan)], dtype=order_dtype)
            return np.rec.array(order_arr)

    # Build actual order using order_nb when available (preferred)
    if _order_nb_available and _order_nb is not None:
        try:
            return _order_nb(size_val, price_val)
        except Exception:
            order_dtype = np.dtype([
                ("size", np.float64),
                ("price", np.float64),
                ("stop_price", np.float64),
                ("limit_price", np.float64),
            ])
            order_arr = np.array([(size_val, price_val, stop_val, limit_val)], dtype=order_dtype)
            return np.rec.array(order_arr)
    else:
        order_dtype = np.dtype([
            ("size", np.float64),
            ("price", np.float64),
            ("stop_price", np.float64),
            ("limit_price", np.float64),
        ])
        order_arr = np.array([(size_val, price_val, stop_val, limit_val)], dtype=order_dtype)
        return np.rec.array(order_arr)
