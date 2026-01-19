import inspect
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


def _call_with_signature(meth, size, size_type, direction, c):
    """Try calling meth with a variety of argument patterns inferred from its signature."""
    price = np.inf
    try:
        sig = inspect.signature(meth)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    # Candidate call patterns (kwargs first, then positional)
    tries = []

    # If method accepts size and price etc
    if 'size' in params and 'price' in params:
        tries.append({'size': size, 'price': price, 'size_type': size_type, 'direction': direction})
    if 'size' in params and 'size_type' in params:
        tries.append({'size': size, 'size_type': size_type})
    if 'size' in params:
        tries.append({'size': size})
    if 'percent' in params or 'pct' in params:
        tries.append({'percent': size})
    if 'value' in params or 'amount' in params:
        # interpret size in percent if between 0 and 1
        val = size
        try:
            if 0 < float(size) <= 1.0 and hasattr(c, 'cash_now'):
                val = float(size) * float(c.cash_now)
        except Exception:
            pass
        tries.append({'value': val})
        tries.append({'amount': val})

    # Positional fallbacks
    tries.append((size, price, size_type, direction))
    tries.append((size, size_type, direction))
    tries.append((size,))

    # Attempt calls
    for call_args in tries:
        try:
            if isinstance(call_args, dict):
                return meth(**call_args)
            else:
                return meth(*call_args)
        except Exception:
            continue

    # If none worked, raise
    raise RuntimeError("Failed to call method with inferred signatures")


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
    Order function for vectorbt.from_order_func (pure Python, NO NUMBA).

    Implements a MACD crossover entry with ATR-based trailing stop.

    Attempts to use OrderContext helper methods (if available) to construct
    orders. Falls back to returning a simple tuple when helpers are absent.
    """
    i = int(c.i)
    pos = float(c.position_now) if c.position_now is not None else 0.0

    # Initialize persistent state on the function object on first call
    if not hasattr(order_func, "_last_entry_i"):
        order_func._last_entry_i = None
        order_func._highest_since_entry = -np.inf

    # Helper: previous index
    prev_i = i - 1

    # Safely extract current and previous indicator values (handle out-of-bounds)
    def safe_get(arr, idx):
        try:
            return arr[int(idx)]
        except Exception:
            return np.nan

    prev_macd = safe_get(macd, prev_i)
    prev_signal = safe_get(signal, prev_i)

    curr_macd = safe_get(macd, i)
    curr_signal = safe_get(signal, i)
    curr_close = safe_get(close, i)
    curr_high = safe_get(high, i)
    curr_atr = safe_get(atr, i)
    curr_sma = safe_get(sma, i)

    def make_order(size: float, size_type: int = 2, direction: int = 1):
        """Construct order using c helpers if available; otherwise return tuple fallback."""
        # Handle explicit no-order
        if np.isnan(size):
            # Try c.no_order or c.no_action
            if hasattr(c, 'no_order') and callable(getattr(c, 'no_order')):
                try:
                    return c.no_order()
                except Exception:
                    pass
            if hasattr(c, 'no_action') and callable(getattr(c, 'no_action')):
                try:
                    return c.no_action()
                except Exception:
                    pass
            # Fallback: tuple indicating no action
            return (np.nan, 0, 0)

        # Try to call multiple candidate methods found on OrderContext
        candidate_names = [
            'order', 'order_percent', 'order_pct', 'order_value', 'order_amount',
            'order_size', 'order_size_pct', 'buy', 'sell', 'close', 'close_position'
        ]

        for name in candidate_names:
            if hasattr(c, name) and callable(getattr(c, name)):
                meth = getattr(c, name)
                try:
                    res = _call_with_signature(meth, size, size_type, direction, c)
                    # Return the raw result and let vectorbt handle it
                    return res
                except Exception:
                    continue

        # As last resort, try calling c.order_nb if present (Python wrapper around Numba)
        if hasattr(c, 'order_nb') and callable(getattr(c, 'order_nb')):
            try:
                res = _call_with_signature(getattr(c, 'order_nb'), size, size_type, direction, c)
                return res
            except Exception:
                pass

        # Fallback: return tuple per prompt
        return (size, size_type, direction)

    # ENTRY logic
    if pos <= 0.0:
        if order_func._last_entry_i is not None:
            order_func._last_entry_i = None
            order_func._highest_since_entry = -np.inf

        if (
            not np.isnan(prev_macd)
            and not np.isnan(prev_signal)
            and not np.isnan(curr_macd)
            and not np.isnan(curr_signal)
            and not np.isnan(curr_sma)
            and not np.isnan(curr_close)
        ):
            macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            price_above_sma = curr_close > curr_sma

            if macd_cross_up and price_above_sma:
                order_func._last_entry_i = i
                order_func._highest_since_entry = curr_high if not np.isnan(curr_high) else curr_close
                return make_order(0.95, size_type=2, direction=1)

        return make_order(np.nan, size_type=0, direction=0)

    # IN POSITION logic
    if order_func._last_entry_i is None:
        order_func._last_entry_i = i
        order_func._highest_since_entry = curr_high if not np.isnan(curr_high) else curr_close

    if not np.isnan(curr_high):
        if not np.isfinite(order_func._highest_since_entry):
            order_func._highest_since_entry = curr_high
        else:
            order_func._highest_since_entry = max(order_func._highest_since_entry, curr_high)

    trailing_stop = np.nan
    if np.isfinite(order_func._highest_since_entry) and (not np.isnan(curr_atr)):
        trailing_stop = order_func._highest_since_entry - trailing_mult * curr_atr

    macd_cross_down = False
    if (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and not np.isnan(curr_macd)
        and not np.isnan(curr_signal)
    ):
        macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

    trail_hit = False
    if not np.isnan(trailing_stop) and not np.isnan(curr_close):
        trail_hit = curr_close < trailing_stop

    if macd_cross_down or trail_hit:
        order_func._last_entry_i = None
        order_func._highest_since_entry = -np.inf
        return make_order(-np.inf, size_type=2, direction=1)

    return make_order(np.nan, size_type=0, direction=0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy using vectorbt indicator wrappers.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input ohlcv index.
    """
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'high' and 'close' columns")

    close_s = ohlcv["close"]
    high_s = ohlcv["high"]

    if "low" in ohlcv.columns:
        low_s = ohlcv["low"]
    else:
        low_s = close_s

    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    return {
        "close": close_s.values,
        "high": high_s.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }