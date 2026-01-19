"""
Trading strategy: MACD crossover entries with ATR-based trailing stops.

Exports:
- compute_indicators
- order_func

Notes:
- No Numba decorators used by this module.
- The order function is written to try to create order objects compatible with
  vectorbt internals. This includes a heuristic that looks for an Order
  namedtuple in vectorbt.portfolio.enums and builds instances accordingly.

State for the order function is kept in module-level variables and reset
when compute_indicators is called so backtests don't leak state across runs.
"""

from collections import namedtuple
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Order namedtuple fallback (tuple subclass with attribute access)
_Order = namedtuple("_Order", ["price", "size", "sl_price", "tp_price"])

# Module-level state used by order_func (reset in compute_indicators)
_ORDER_STATE: Dict[str, Any] = {
    "in_position": False,
    "position_size": 0.0,
    "highest_since_entry": -np.inf,
}

# Optional factory to create order objects compatible with vectorbt internals
_ORDER_FACTORY: Optional[Callable[..., Any]] = None

# Helper references to the indicator arrays - set in compute_indicators
_CLOSE: np.ndarray = np.array([])
_HIGH: np.ndarray = np.array([])
_MACD: np.ndarray = np.array([])
_SIGNAL: np.ndarray = np.array([])
_ATR: np.ndarray = np.array([])
_SMA: np.ndarray = np.array([])


def _is_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float, np.floating, np.integer))
    except Exception:
        return False


def _make_order_factory_from_enum(OrderTuple: Any) -> Optional[Callable[..., Any]]:
    """Given a namedtuple-like Order type (from vectorbt), build a factory that
    constructs Order instances with sensible default values based on field names.
    """
    # Must have _fields
    if not hasattr(OrderTuple, "_fields"):
        return None

    fields = list(getattr(OrderTuple, "_fields"))

    def factory(price: float, size: float, sl_price: float, tp_price: float):
        vals: list = []
        for f in fields:
            fname = f.lower()
            if fname == "price":
                vals.append(float(price))
            elif fname == "size":
                vals.append(float(size))
            elif "stop" in fname and "price" in fname:
                vals.append(float(sl_price) if not np.isnan(sl_price) else 0.0)
            elif ("sl" == fname) or ("sl_" in fname) or ("slprice" in fname):
                vals.append(float(sl_price) if not np.isnan(sl_price) else 0.0)
            elif "tp" in fname or "take" in fname:
                vals.append(float(tp_price) if not np.isnan(tp_price) else 0.0)
            elif "slippage" in fname:
                vals.append(0.0)
            elif "max" in fname:
                # max fields must be > 0
                vals.append(max(1.0, abs(float(size)) if _is_number(size) else 1.0))
            elif "pct" in fname or "%" in fname:
                vals.append(0.0)
            elif "type" in fname or "unit" in fname:
                vals.append(0)
            elif "is_" in fname or fname in ("reduce_only", "post_only", "hidden"):
                vals.append(False)
            else:
                # default numeric
                vals.append(0)
        try:
            return OrderTuple(*vals)
        except Exception:
            return None

    return factory


def _discover_order_factory() -> Optional[Callable[..., Any]]:
    """Try to discover a suitable order factory in the vectorbt package.

    Preference order:
      1. vbt.portfolio.enums.Order (most likely namedtuple describing order fields)
      2. functions in vbt.portfolio.nb that create orders
      3. fallback to None
    """
    try:
        import vectorbt as vbt
    except Exception:
        return None

    # 1) Try enums.Order
    try:
        if hasattr(vbt.portfolio, "enums") and hasattr(vbt.portfolio.enums, "Order"):
            OrderTuple = getattr(vbt.portfolio.enums, "Order")
            factory = _make_order_factory_from_enum(OrderTuple)
            if factory is not None:
                return factory
    except Exception:
        # ignore and continue
        pass

    # 2) Try functions in vbt.portfolio.nb (order_nb, no_order_nb)
    try:
        if hasattr(vbt.portfolio, "nb"):
            nb_mod = vbt.portfolio.nb
            for cand in ("order_nb", "no_order_nb", "make_order", "create_order"):
                if hasattr(nb_mod, cand):
                    attr = getattr(nb_mod, cand)
                    # Test calling with plausible args
                    for args in ((1.0, 1.0, float("nan"), float("nan")), (1.0, 1.0), ()):  # try common signatures
                        try:
                            res = attr(*args)
                        except Exception:
                            continue
                        if hasattr(res, "price") and _is_number(getattr(res, "price")):
                            if hasattr(res, "size") and _is_number(getattr(res, "size")):
                                # Build a small wrapper around this function
                                def _wrap(f):
                                    def _wrapped(p, s, sl, tp):
                                        try:
                                            return f(p, s, sl, tp)
                                        except TypeError:
                                            try:
                                                return f(p, s)
                                            except Exception:
                                                return f()
                                    return _wrapped

                                return _wrap(attr)
    except Exception:
        pass

    # 3) fallback
    return None


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Returns a dict with numpy arrays for keys: close, high, macd, signal, atr, sma.

    All arrays are 1-D float64 arrays aligned with ohlcv.index.
    """
    global _ORDER_STATE, _CLOSE, _HIGH, _MACD, _SIGNAL, _ATR, _SMA, _ORDER_FACTORY

    # Validate input
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # MACD (EMA_fast - EMA_slow) and signal line (EMA of MACD)
    # Use ewm with adjust=False (common practice)
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd = (ema_fast - ema_slow).astype(float)
    signal = macd.ewm(span=macd_signal, adjust=False).mean().astype(float)

    # SMA trend filter
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean().astype(float)

    # ATR: True Range then rolling mean (simple ATR). Use min_periods=atr_period to produce NaNs for warmup
    prev_close = close_s.shift(1)
    tr1 = (high_s - low_s).abs()
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().astype(float)

    # Convert to numpy arrays
    close_arr = close_s.values.astype(float)
    high_arr = high_s.values.astype(float)
    macd_arr = macd.values.astype(float)
    signal_arr = signal.values.astype(float)
    atr_arr = atr.values.astype(float)
    sma_arr = sma.values.astype(float)

    # Reset order state for a fresh backtest
    _ORDER_STATE = {
        "in_position": False,
        "position_size": 0.0,
        "highest_since_entry": -np.inf,
    }

    # Store references for order_func to use (not strictly necessary but useful)
    _CLOSE = close_arr
    _HIGH = high_arr
    _MACD = macd_arr
    _SIGNAL = signal_arr
    _ATR = atr_arr
    _SMA = sma_arr

    # Discover order factory if available
    _ORDER_FACTORY = _discover_order_factory()

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }


def _extract_index_from_ctx(order_ctx: Any) -> int:
    """Robustly extract integer index 'i' from the order context or scalar.

    Vectorbt's from_order_func calls the order function with a context-like
    object when simulate runs in its internal loops. In other usages an
    integer index might be passed directly. Accept a few common attribute
    names used internally.
    """
    # Direct integer
    if isinstance(order_ctx, (int, np.integer)):
        return int(order_ctx)

    # Numpy scalar
    if isinstance(order_ctx, np.ndarray) and order_ctx.shape == ():
        try:
            return int(order_ctx.tolist())
        except Exception:
            pass

    # Try common attributes
    for attr in ("i", "index", "idx"):
        if hasattr(order_ctx, attr):
            val = getattr(order_ctx, attr)
            try:
                return int(val)
            except Exception:
                continue

    # As a last resort, try to convert to int
    try:
        return int(order_ctx)
    except Exception as e:
        raise ValueError("Unable to extract integer index from order context") from e


def _make_order_from_factory(factory: Optional[Callable[..., Any]], price: float, size: float, sl_price: float, tp_price: float) -> Any:
    """Use discovered factory to build an order-like object. Try several signatures."""
    # Try using the factory if discovered
    if factory is not None:
        for args in ((price, size, sl_price, tp_price), (price, size), ()):  # try common signatures
            try:
                res = factory(*args)
            except TypeError:
                continue
            except Exception:
                # If the factory exists but calling fails, we skip to fallback
                break
            # Ensure the created object looks like an order
            if hasattr(res, "price") and _is_number(getattr(res, "price")):
                if hasattr(res, "size") and _is_number(getattr(res, "size")):
                    # Ensure slippage and max_size are OK if present
                    try:
                        if hasattr(res, "slippage"):
                            s = getattr(res, "slippage")
                            if (not _is_number(s)) or np.isnan(s):
                                try:
                                    setattr(res, "slippage", 0.0)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        if hasattr(res, "max_size"):
                            m = getattr(res, "max_size")
                            if (not _is_number(m)) or (m <= 0):
                                try:
                                    setattr(res, "max_size", max(1.0, abs(float(size))))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return res
            if isinstance(res, tuple) and len(res) >= 2 and _is_number(res[0]) and _is_number(res[1]):
                return res
    # Fallback to namedtuple
    return _Order(price, size, sl_price, tp_price)


def order_func(
    order_ctx: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Any:
    """Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    Parameters are expected to be passed exactly in this order by the caller:
    (order_ctx, close, high, macd, signal, atr, sma, trailing_mult)

    Returns an order-like object. Prefer using a discovered factory when
    available so the returned object is compatible with the compiled simulator.

    Notes:
    - Long-only.
    - Uses simple "buy 1 unit on entry, sell all units on exit" sizing.
    """
    global _ORDER_STATE, _ORDER_FACTORY

    # Extract integer index
    i = _extract_index_from_ctx(order_ctx)

    # Defensive: prefer the arrays passed in to the globals so function works when
    # called directly with arrays. But keep the globals for state only.
    close_arr = close
    high_arr = high
    macd_arr = macd
    signal_arr = signal
    atr_arr = atr
    sma_arr = sma

    # Current values
    price = float(close_arr[i])
    high_price = float(high_arr[i])
    macd_val = float(macd_arr[i])
    signal_val = float(signal_arr[i])
    atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else float("nan")
    sma_val = float(sma_arr[i]) if not np.isnan(sma_arr[i]) else float("nan")

    # Previous values (for crossover detection)
    if i > 0:
        macd_prev = float(macd_arr[i - 1])
        signal_prev = float(signal_arr[i - 1])
    else:
        macd_prev = float("nan")
        signal_prev = float("nan")

    in_position = bool(_ORDER_STATE["in_position"])

    order_size = 0.0

    # Entry detection: MACD crosses above signal AND price > SMA
    entry = False
    if not in_position:
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and macd_prev <= signal_prev
            and macd_val > signal_val
        ):
            # Check SMA condition
            if not np.isnan(sma_val) and price > sma_val:
                entry = True

    # Update highest_since_entry if currently in position (track intrabar highs)
    if in_position:
        # initialize highest value if it was -inf
        if _ORDER_STATE["highest_since_entry"] == -np.inf:
            _ORDER_STATE["highest_since_entry"] = high_price
        else:
            if high_price > _ORDER_STATE["highest_since_entry"]:
                _ORDER_STATE["highest_since_entry"] = high_price

    # Trailing stop calculation
    trailing_stop_hit = False
    if in_position:
        highest = _ORDER_STATE["highest_since_entry"]
        if not np.isnan(atr_val) and highest is not None and highest != -np.inf:
            stop_price = highest - (trailing_mult * atr_val)
            # Only consider the stop if it is a finite number
            if not np.isnan(stop_price) and price < stop_price:
                trailing_stop_hit = True

    # Exit detection via MACD cross below
    exit_by_macd = False
    if in_position:
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and macd_prev >= signal_prev
            and macd_val < signal_val
        ):
            exit_by_macd = True

    # Decide orders
    if entry:
        # Enter long: buy 1 unit
        order_size = 1.0
        # Update state assuming order will be executed - vectorbt will execute if cash is available.
        _ORDER_STATE["in_position"] = True
        _ORDER_STATE["position_size"] = float(order_size)
        _ORDER_STATE["highest_since_entry"] = high_price

    elif in_position and (exit_by_macd or trailing_stop_hit):
        # Exit: sell all units we have
        order_size = -float(_ORDER_STATE.get("position_size", 1.0) or 1.0)
        # Reset state
        _ORDER_STATE["in_position"] = False
        _ORDER_STATE["position_size"] = 0.0
        _ORDER_STATE["highest_since_entry"] = -np.inf

    # Create order-like object using discovered factory when possible
    return _make_order_from_factory(_ORDER_FACTORY, float(price), float(order_size), 0.0, 0.0)
