import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

# Import Order namedtuple and helpers from vectorbt
from vectorbt.portfolio import enums as vbt_enums


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy and return them as numpy arrays.

    Returns keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close = ohlcv["close"].astype(float).copy()
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()

    # MACD (EMA-based)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR (Wilder's smoothing via EMA)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
    }


def _flex_get(a: Any, idx: int, col: int) -> float:
    """Get element for flexible-indexed arrays used by vectorbt order func.

    Supports 0-dim, 1-dim and 2-dim arrays (numpy / pandas structures).
    Falls back to NaN when indexing is out-of-bounds.
    """
    arr = np.asarray(a)
    if arr.ndim == 0:
        try:
            return float(arr)
        except Exception:
            return float(np.nan)
    if arr.ndim == 1:
        if 0 <= idx < arr.shape[0]:
            return float(arr[idx])
        return float(np.nan)
    if arr.ndim == 2:
        # shape (n_rows, n_cols)
        if 0 <= idx < arr.shape[0] and 0 <= col < arr.shape[1]:
            return float(arr[idx, col])
        # try flattened fallback
        try:
            return float(arr[idx])
        except Exception:
            return float(np.nan)
    # fallback
    try:
        return float(arr.reshape(arr.shape[0], -1)[idx, 0])
    except Exception:
        return float(np.nan)


def _flex_get_range(a: Any, start: int, end: int, col: int) -> np.ndarray:
    """Get slice between start and end (inclusive of start, exclusive of end) for flexible arrays."""
    arr = np.asarray(a)
    if arr.ndim == 0:
        return np.array([float(arr)])
    if arr.ndim == 1:
        return arr[start:end]
    if arr.ndim == 2:
        # return column slice
        return arr[start:end, col]
    # fallback
    return np.asarray(arr)[start:end]


def order_func(*args: Any) -> Any:
    """
    Dual-mode order function.

    Modes supported:
    1) Numba-style (used by vbt.Portfolio.from_order_func when use_numba=False):
       Called as order_func(order_ctx, close, high, macd, signal, atr, sma, trailing_mult)
       Should return either `vectorbt.portfolio.enums.Order` or `vectorbt.portfolio.enums.NoOrder`.

    2) Direct-signal mode (used by unit tests):
       Called as order_func(close, high, macd, signal, atr, sma, trailing_mult)
       Returns tuple (entries, exits) as boolean numpy arrays of length len(close).

    The function is carefully implemented to be causal (no lookahead).
    """
    # Detect Numba-style call: first arg has attribute 'i' (OrderContext)
    if len(args) >= 1 and hasattr(args[0], "i"):
        # Unpack order context style: order_ctx + order args
        order_ctx = args[0]
        if len(args) < 8:
            raise ValueError("order_func in nb-style expects: (order_ctx, close, high, macd, signal, atr, sma, trailing_mult)")
        close_arr = args[1]
        high_arr = args[2]
        macd_arr = args[3]
        signal_arr = args[4]
        atr_arr = args[5]
        sma_arr = args[6]
        trailing_mult = args[7]

        # Current index and column
        i = int(order_ctx.i)
        col = int(order_ctx.col)

        # Helper to access previous values safely
        def get_elem(a, idx_offset: int = 0) -> float:
            idx = i + idx_offset
            return _flex_get(a, idx, col)

        macd_curr = get_elem(macd_arr, 0)
        signal_curr = get_elem(signal_arr, 0)
        macd_prev = get_elem(macd_arr, -1) if i > 0 else float(np.nan)
        signal_prev = get_elem(signal_arr, -1) if i > 0 else float(np.nan)
        price_curr = get_elem(close_arr, 0)
        sma_curr = get_elem(sma_arr, 0)
        atr_curr = get_elem(atr_arr, 0)

        # Current position (from context)
        position_now = float(order_ctx.position_now)

        # Entry: MACD bullish cross and price > SMA
        if position_now == 0:
            if (
                not np.isnan(macd_curr) and not np.isnan(signal_curr) and
                not np.isnan(macd_prev) and not np.isnan(signal_prev)
                and macd_curr > signal_curr and macd_prev <= signal_prev
            ):
                if not np.isnan(sma_curr) and not np.isnan(price_curr) and price_curr > sma_curr:
                    # Enter long: buy with all cash (size = +inf, price = +inf -> replaced by close)
                    return vbt_enums.Order(
                        size=np.inf,
                        price=np.inf,
                        direction=vbt_enums.Direction.LongOnly,
                    )
            return vbt_enums.NoOrder

        # If in position -> check trailing stop and MACD bearish cross
        # Trailing stop: price < (highest_since_entry - trailing_mult * ATR)
        # Compute entry index from pos_record (if any)
        entry_idx = -1
        try:
            # pos_record_now is a numpy record with fields like 'entry_idx'
            entry_idx = int(getattr(order_ctx.pos_record_now, 'entry_idx', -1))
        except Exception:
            try:
                entry_idx = int(order_ctx.pos_record_now['entry_idx'])
            except Exception:
                entry_idx = -1

        # Trailing stop check
        if entry_idx is not None and entry_idx >= 0 and not np.isnan(atr_curr):
            # highest high since entry
            hi_slice = _flex_get_range(high_arr, entry_idx, i + 1, col)
            # ensure we can compute a max
            if len(hi_slice) > 0:
                try:
                    highest_since = float(np.nanmax(np.asarray(hi_slice).astype(float)))
                except Exception:
                    highest_since = float(np.nan)
                if not np.isnan(highest_since) and not np.isnan(atr_curr):
                    threshold = highest_since - float(trailing_mult) * float(atr_curr)
                    if not np.isnan(price_curr) and price_curr < threshold:
                        # Exit: close long position (size = -inf with LongOnly will close)
                        return vbt_enums.Order(
                            size=-np.inf,
                            price=np.inf,
                            direction=vbt_enums.Direction.LongOnly,
                        )

        # MACD bearish cross exit
        if (
            not np.isnan(macd_curr) and not np.isnan(signal_curr) and
            not np.isnan(macd_prev) and not np.isnan(signal_prev)
            and macd_curr < signal_curr and macd_prev >= signal_prev
        ):
            return vbt_enums.Order(
                size=-np.inf,
                price=np.inf,
                direction=vbt_enums.Direction.LongOnly,
            )

        return vbt_enums.NoOrder

    # Direct (vectorized) mode: produce entry/exit boolean arrays
    # Supports calling conventions:
    #   order_func(close, high, macd, signal, atr, sma, trailing_mult)
    #   or order_func(price, close, high, macd, signal, atr, sma, trailing_mult)
    if len(args) == 7:
        close_arr, high_arr, macd_arr, signal_arr, atr_arr, sma_arr, trailing_mult = args
    elif len(args) == 8:
        # possibly price passed first - ignore it
        _, close_arr, high_arr, macd_arr, signal_arr, atr_arr, sma_arr, trailing_mult = args
    else:
        raise ValueError("order_func expects either nb-style or direct-call with 7/8 args")

    close = np.asarray(close_arr, dtype=float)
    high = np.asarray(high_arr, dtype=float)
    macd = np.asarray(macd_arr, dtype=float)
    signal = np.asarray(signal_arr, dtype=float)
    atr = np.asarray(atr_arr, dtype=float)
    sma = np.asarray(sma_arr, dtype=float)

    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    in_position = False
    highest_since_entry = -np.inf

    for t in range(1, n):
        # Entry
        if not in_position:
            if (
                not np.isnan(macd[t]) and not np.isnan(signal[t]) and
                not np.isnan(macd[t - 1]) and not np.isnan(signal[t - 1]) and
                macd[t] > signal[t] and macd[t - 1] <= signal[t - 1]
            ):
                if not np.isnan(sma[t]) and not np.isnan(close[t]) and close[t] > sma[t]:
                    entries[t] = True
                    in_position = True
                    highest_since_entry = high[t] if not np.isnan(high[t]) else close[t]
        else:
            # update highest
            if not np.isnan(high[t]) and high[t] > highest_since_entry:
                highest_since_entry = high[t]
            # trailing stop
            if not np.isnan(atr[t]) and np.isfinite(highest_since_entry):
                threshold = highest_since_entry - float(trailing_mult) * atr[t]
                if not np.isnan(close[t]) and close[t] < threshold:
                    exits[t] = True
                    in_position = False
                    highest_since_entry = -np.inf
                    continue
            # macd bearish cross
            if (
                not np.isnan(macd[t]) and not np.isnan(signal[t]) and
                not np.isnan(macd[t - 1]) and not np.isnan(signal[t - 1]) and
                macd[t] < signal[t] and macd[t - 1] >= signal[t - 1]
            ):
                exits[t] = True
                in_position = False
                highest_since_entry = -np.inf

    return entries, exits
