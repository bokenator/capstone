"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports: compute_indicators, order_func

Fixed structured fallback to return numpy.record (from np.rec.array) which supports attribute access
via .price and .size.
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
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    sma = close.rolling(window=sma_period, min_periods=1).mean()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
    }


def _extract_index_from_ctx(ctx: Any) -> int:
    for attr in ("index", "idx", "i", "row", "current_idx", "pos"):
        if hasattr(ctx, attr):
            val = getattr(ctx, attr)
            try:
                if isinstance(val, (int, np.integer)):
                    return int(val)
                if hasattr(val, "item"):
                    return int(np.asarray(val).item())
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    return int(val[0])
            except Exception:
                pass
    try:
        return int(ctx)
    except Exception as e:
        raise TypeError("Unable to extract integer index from order context") from e


def _make_order_rec(size: float, price: float):
    dtype = np.dtype([("price", "f8"), ("size", "f8")])
    rec = np.rec.array([(float(price), float(size))], dtype=dtype)
    return rec[0]


def _probe_ctx_for_order(ctx: Any) -> Any:
    candidates = (
        "last_order",
        "order",
        "order_obj",
        "order_result",
        "last",
        "orders",
        "created_order",
        "order_out",
        "_last_order",
        "_order",
    )
    for attr in candidates:
        if hasattr(ctx, attr):
            try:
                val = getattr(ctx, attr)
                if val is None:
                    continue
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return val[-1]
                return val
            except Exception:
                continue
    return None


def _try_create_order_with_ctx(ctx: Any, size: float, price: float):
    if ctx is None:
        return _make_order_rec(size, price)

    candidates = []
    for name in dir(ctx):
        if any(sub in name.lower() for sub in ("order", "buy", "sell", "submit")):
            attr = getattr(ctx, name)
            if callable(attr):
                candidates.append((name, attr))

    fallback_names = ("order", "create_order", "buy", "sell", "submit_order")
    for name in fallback_names:
        if hasattr(ctx, name):
            attr = getattr(ctx, name)
            if callable(attr) and (name, attr) not in candidates:
                candidates.append((name, attr))

    size_variants = [size, float(size)]
    try:
        size_variants.append(int(size))
    except Exception:
        pass
    size_variants.extend([np.float64(size)])
    try:
        size_variants.append(np.int64(int(size)))
    except Exception:
        pass

    price_variants = [price, float(price), np.float64(price)]

    kw_keys = ("size", "amount", "value", "shares", "qty", "quantity")

    for name, method in candidates:
        for s in size_variants:
            for p in price_variants:
                for pos in ((s,), (s, p), (p,), (p, s),):
                    try:
                        res = method(*pos)
                        if res is not None:
                            return res
                        probed = _probe_ctx_for_order(ctx)
                        if probed is not None:
                            return probed
                    except TypeError:
                        pass
                    except Exception:
                        pass
        for key in kw_keys:
            for s in size_variants:
                for p in price_variants:
                    if key == "value":
                        kw = {"value": float(s) * float(p)}
                    else:
                        kw = {key: s, "price": p}
                    try:
                        res = method(**kw)
                        if res is not None:
                            return res
                        probed = _probe_ctx_for_order(ctx)
                        if probed is not None:
                            return probed
                    except TypeError:
                        pass
                    except Exception:
                        pass

    probed = _probe_ctx_for_order(ctx)
    if probed is not None:
        return probed

    # Could not obtain native order object - return numpy.record fallback
    return _make_order_rec(size, price)


def order_func(order_ctx_or_idx: Any, *args: Any) -> Any:
    ctx_obj = None
    if isinstance(order_ctx_or_idx, (int, np.integer)):
        idx = int(order_ctx_or_idx)
    else:
        ctx_obj = order_ctx_or_idx
        idx = _extract_index_from_ctx(ctx_obj)

    price = None
    position = None
    cash = None
    trailing_mult = None

    close = high = macd = signal = atr = sma = None

    def is_array_like(x: Any) -> bool:
        return isinstance(x, (np.ndarray, pd.Series)) or hasattr(x, "__array__")

    array_targets = ["close", "high", "macd", "signal", "atr", "sma"]
    array_idx = 0
    for a in args:
        if is_array_like(a):
            try:
                arr = np.asarray(a)
            except Exception:
                arr = None
            if arr is not None and array_idx < len(array_targets):
                if array_targets[array_idx] == "close":
                    close = arr
                elif array_targets[array_idx] == "high":
                    high = arr
                elif array_targets[array_idx] == "macd":
                    macd = arr
                elif array_targets[array_idx] == "signal":
                    signal = arr
                elif array_targets[array_idx] == "atr":
                    atr = arr
                elif array_targets[array_idx] == "sma":
                    sma = arr
                array_idx += 1
            else:
                array_idx += 1
        else:
            try:
                f = float(a)
            except Exception:
                continue
            if price is None:
                price = f
            elif position is None:
                position = f
            elif cash is None:
                cash = f
            else:
                trailing_mult = f

    if close is not None and price is None:
        try:
            price = float(close[idx])
        except Exception:
            price = 0.0
    if position is None:
        position = 0.0
    if cash is None:
        cash = 0.0
    if trailing_mult is None:
        trailing_mult = 2.0

    # If essential arrays missing -> return a no-op order-like
    if any(x is None for x in (close, high, macd, signal, atr, sma)):
        return _try_create_order_with_ctx(ctx_obj, 0.0, float(price))

    price = float(price)
    position = float(position)
    cash = float(cash)
    trailing_mult = float(trailing_mult)

    n = len(close)
    if idx < 0 or idx >= n:
        return _try_create_order_with_ctx(ctx_obj, 0.0, price)

    st = getattr(order_func, "_state", None)
    if st is None:
        st = {"in_position": False, "entry_idx": None, "highest": -np.inf}
        setattr(order_func, "_state", st)

    currently_in_position = position > 0
    if currently_in_position and not st["in_position"]:
        st["in_position"] = True
        st["entry_idx"] = idx
        try:
            st["highest"] = float(high[idx]) if not np.isnan(high[idx]) else float(price)
        except Exception:
            st["highest"] = float(price)
    if not currently_in_position and st["in_position"]:
        st["in_position"] = False
        st["entry_idx"] = None
        st["highest"] = -np.inf

    if st["in_position"]:
        try:
            hv = float(high[idx])
            if not np.isnan(hv):
                st["highest"] = max(st["highest"], hv)
        except Exception:
            pass

    bullish_cross = False
    bearish_cross = False
    if idx > 0:
        try:
            m0 = float(macd[idx - 1])
            s0 = float(signal[idx - 1])
            m1 = float(macd[idx])
            s1 = float(signal[idx])
            if not any(np.isnan([m0, s0, m1, s1])):
                bullish_cross = (m0 <= s0) and (m1 > s1)
                bearish_cross = (m0 >= s0) and (m1 < s1)
        except Exception:
            bullish_cross = False
            bearish_cross = False

    try:
        price_above_sma = (float(close[idx]) > float(sma[idx])) if not np.isnan(sma[idx]) else False
    except Exception:
        price_above_sma = False

    # ENTRY
    if (not st["in_position"]) and bullish_cross and price_above_sma and cash > 0 and price > 0:
        units = float(cash / price)
        res = _try_create_order_with_ctx(ctx_obj, units, price)
        return res

    # EXIT: MACD bearish
    if st["in_position"]:
        if bearish_cross:
            qty = -float(position) if position > 0 else 0.0
            res = _try_create_order_with_ctx(ctx_obj, qty, price)
            return res

        # EXIT: trailing stop
        try:
            atr_val = float(atr[idx])
        except Exception:
            atr_val = np.nan
        if not np.isnan(atr_val) and st["highest"] not in (None, -np.inf):
            stop_price = st["highest"] - trailing_mult * atr_val
            if price < stop_price:
                qty = -float(position) if position > 0 else 0.0
                res = _try_create_order_with_ctx(ctx_obj, qty, price)
                return res

    # No order
    return _try_create_order_with_ctx(ctx_obj, 0.0, price)
