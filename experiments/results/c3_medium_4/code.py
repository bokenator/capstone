# Complete strategy implementation: MACD crossover entries with ATR-based trailing stops
# Exports:
# - compute_indicators(ohlcv: pd.DataFrame, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
#                      sma_period: int = 50, atr_period: int = 14) -> pd.DataFrame
# - order_func(ctx, *args) -> order object (via nb.order_nb) or tuple fallback

from typing import Dict, Any

import inspect
import numpy as np
import pandas as pd

# Attempt to prepare a factory around vbt.portfolio.nb.order_nb / no_order_nb if available
_ORDER_NB_FACTORY = None
_NO_ORDER_NB = None
try:
    import vectorbt as vbt
    nb_mod = getattr(vbt.portfolio, "nb", None)
    if nb_mod is not None:
        if hasattr(nb_mod, "order_nb"):
            order_nb_fn = getattr(nb_mod, "order_nb")
            order_nb_sig = inspect.signature(order_nb_fn)
            order_nb_params = [p.name for p in order_nb_sig.parameters.values()]

            def _order_nb_factory(size: float):
                # Build kwargs for order_nb using parameter name heuristics
                kwargs = {}
                for pname in order_nb_params:
                    pl = pname.lower()
                    if "size" in pl and "size_type" not in pl and "max" not in pl:
                        kwargs[pname] = float(size)
                    elif "max" in pl:
                        # ensure positive max
                        kwargs[pname] = float(abs(size)) if float(size) != 0 else 1.0
                    elif "granularity" in pl or "gran" in pl:
                        kwargs[pname] = float("nan")
                    elif "min" in pl:
                        kwargs[pname] = 1e-9
                    elif "price" in pl:
                        # market order marker
                        kwargs[pname] = float(np.inf)
                    elif "size_type" in pl:
                        kwargs[pname] = 0
                    elif "price_type" in pl:
                        kwargs[pname] = 0
                    elif "allow" in pl:
                        kwargs[pname] = True
                    elif "side" in pl:
                        kwargs[pname] = 1 if size > 0 else (-1 if size < 0 else 0)
                    else:
                        # generic filler
                        kwargs[pname] = 0
                return order_nb_fn(**kwargs)

            _ORDER_NB_FACTORY = _order_nb_factory
        if hasattr(nb_mod, "no_order_nb"):
            no_order_nb_fn = getattr(nb_mod, "no_order_nb")

            def _no_order_nb():
                return no_order_nb_fn()

            _NO_ORDER_NB = _no_order_nb
except Exception:
    # If vectorbt not available at import time or structure differs, leave factories None
    _ORDER_NB_FACTORY = None
    _NO_ORDER_NB = None


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute technical indicators required by the strategy.

    Returns a pandas DataFrame with columns: ['close', 'high', 'macd', 'signal', 'atr', 'sma']
    The index is preserved from the input ohlcv.

    All calculations are causal (no lookahead).
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD: EMA(fast) - EMA(slow)
    # Use pandas ewm with adjust=False (recursive) to avoid lookahead
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # ATR: True Range then Wilder's smoothing (using ewm with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing approximated with ewm(adjust=False, alpha=1/atr_period)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    indicators = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "macd": macd,
            "signal": signal,
            "atr": atr,
            "sma": sma,
        },
        index=ohlcv.index,
    )

    return indicators


def _extract_idx_from_ctx(ctx) -> int:
    """
    Robustly extract integer index from vectorbt OrderContext or plain int.
    """
    # Direct int-like
    if isinstance(ctx, (int, np.integer)):
        return int(ctx)

    # Common attribute names to try
    for attr in ("i", "idx", "index", "t", "timestamp_idx", "current_index", "current_idx"):
        if hasattr(ctx, attr):
            try:
                val = getattr(ctx, attr)
                return int(val)
            except Exception:
                pass

    # Methods that may exist
    for method_name in ("get_current_index", "get_current_tick", "current_index", "get_idx"):
        if hasattr(ctx, method_name):
            try:
                method = getattr(ctx, method_name)
                val = method() if callable(method) else method
                return int(val)
            except Exception:
                pass

    # Fallback: try to convert to int directly
    try:
        return int(ctx)
    except Exception:
        # As last resort, raise informative error
        raise TypeError("Could not extract integer index from order context")


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _is_valid_order_obj(obj) -> bool:
    # We consider an object valid if it exposes a numeric 'price' attribute
    return hasattr(obj, "price") and isinstance(getattr(obj, "price"), (int, float, np.floating))


def _make_order_from_ctx(ctx: Any, size: float):
    """
    Try to create an order using vbt.portfolio.nb.order_nb if available (preferred),
    or methods on the order context. If none available, return a simple tuple.
    """
    # If asking to place no order, prefer the no_order factory if available
    if float(size) == 0.0 and _NO_ORDER_NB is not None:
        try:
            res = _NO_ORDER_NB()
            if _is_valid_order_obj(res):
                return res
        except Exception:
            pass

    # Prefer nb factory if available
    if _ORDER_NB_FACTORY is not None:
        try:
            res = _ORDER_NB_FACTORY(size)
            if _is_valid_order_obj(res):
                return res
        except Exception:
            pass

    if size is None:
        size = 0.0

    # Candidate factory method names to try
    candidate_names = [
        "create_order", "create_market_order", "create_limit_order", "order", "market_order",
        "market", "limit_order", "buy", "sell", "enter", "exit", "close", "close_all",
        "open", "submit_order", "place_order", "order_nb", "no_order_nb", "no_order",
        "order_target", "order_target_percent", "order_target_value", "order_size",
    ]

    # Try calling context factory methods with common argument names and accept only
    # results that expose a numeric 'price' attribute (likely the proper order object)
    for name in candidate_names:
        if hasattr(ctx, name):
            method = getattr(ctx, name)
            # Try a few common signatures
            for call in ( ((), {"size": size}), ((), {"amount": size}), ((size,), {}), ((), {}) ):
                args_call, kwargs_call = call
                try:
                    res = _try_call(method, *args_call, **kwargs_call)
                    if res is not None and _is_valid_order_obj(res):
                        return res
                except Exception:
                    pass

    # Final fallback: return a simple tuple (size,). The wrapper may or may not accept this.
    return (float(size),)


def order_func(ctx: Any, *args: Any):
    """
    Order function used by vectorbt.Portfolio.from_order_func (use_numba=False).

    Signature is flexible to accommodate vectorbt's OrderContext as the first argument.
    The numeric arrays expected (close, high, macd, signal, atr, sma, trailing_mult) are taken
    from the last arguments passed to this function (this handles possible pre-segment outputs).

    Returns either an order object created by nb.order_nb (preferred) or a simple tuple fallback.
    """
    # Expected trailing args count (close, high, macd, signal, atr, sma, trailing_mult)
    EXPECTED = 7

    if len(args) < EXPECTED:
        tail = args
    else:
        tail = args[-EXPECTED:]

    try:
        close_arr, high_arr, macd_arr, signal_arr, atr_arr, sma_arr, trailing_mult = tail
    except Exception:
        close_arr = np.asarray([])
        high_arr = np.asarray([])
        macd_arr = np.asarray([])
        signal_arr = np.asarray([])
        atr_arr = np.asarray([])
        sma_arr = np.asarray([])
        trailing_mult = float(2.0)

    close_arr = np.asarray(close_arr)
    high_arr = np.asarray(high_arr)
    macd_arr = np.asarray(macd_arr)
    signal_arr = np.asarray(signal_arr)
    atr_arr = np.asarray(atr_arr)
    sma_arr = np.asarray(sma_arr)

    # Extract current index from ctx
    i = _extract_idx_from_ctx(ctx)
    n = len(close_arr)

    # Initialize persistent state on first call or when called with a reset (i == 0)
    if not hasattr(order_func, "_state") or i == 0:
        order_func._state = {
            "in_position": False,
            "entry_idx": None,
            "highest": -np.inf,
            "last_idx": -1,
        }

    state = order_func._state

    # Reset state if index not increasing (e.g., new run or truncated data)
    if i <= state.get("last_idx", -1):
        state = {
            "in_position": False,
            "entry_idx": None,
            "highest": -np.inf,
            "last_idx": -1,
        }
        order_func._state = state

    state["last_idx"] = i

    # Bounds check
    if i < 0 or i >= n:
        return _make_order_from_ctx(ctx, 0.0)

    # Safe getter
    def safe_get(arr: np.ndarray, j: int) -> float:
        try:
            v = float(arr[j])
        except Exception:
            v = float("nan")
        return v

    curr_close = safe_get(close_arr, i)
    curr_high = safe_get(high_arr, i)
    curr_macd = safe_get(macd_arr, i)
    curr_signal = safe_get(signal_arr, i)
    curr_atr = safe_get(atr_arr, i)
    curr_sma = safe_get(sma_arr, i)

    # Detect MACD crossings using previous bar (causal)
    cross_up = False
    cross_down = False
    if i > 0:
        prev_macd = safe_get(macd_arr, i - 1)
        prev_signal = safe_get(signal_arr, i - 1)
        if (not np.isnan(prev_macd)) and (not np.isnan(prev_signal)) and (not np.isnan(curr_macd)) and (not np.isnan(curr_signal)):
            cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

    # If not currently in a position, check entry
    if not state["in_position"]:
        if cross_up and (not np.isnan(curr_sma)) and (not np.isnan(curr_close)) and (curr_close > curr_sma):
            state["in_position"] = True
            state["entry_idx"] = i
            state["highest"] = curr_high if not np.isnan(curr_high) else curr_close
            # Create order via nb factory if possible
            return _make_order_from_ctx(ctx, 1.0)
        else:
            return _make_order_from_ctx(ctx, 0.0)

    # If in position, update highest
    if not np.isnan(curr_high) and curr_high > state["highest"]:
        state["highest"] = curr_high

    # Compute trailing stop
    trailing_stop = -np.inf
    if (not np.isnan(state["highest"])) and (not np.isnan(curr_atr)):
        trailing_stop = state["highest"] - float(trailing_mult) * curr_atr

    # Exit conditions
    exited = False
    if cross_down:
        exited = True
    elif (not np.isnan(trailing_stop)) and (not np.isnan(curr_close)) and (curr_close < trailing_stop):
        exited = True

    if exited:
        state["in_position"] = False
        state["entry_idx"] = None
        state["highest"] = -np.inf
        return _make_order_from_ctx(ctx, 0.0)

    return _make_order_from_ctx(ctx, 1.0)
