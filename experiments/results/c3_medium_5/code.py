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
    Compute indicators required by the MACD + ATR trailing stop strategy.

    Returns a dictionary with numpy arrays for keys:
    - macd: MACD line
    - signal: MACD signal line
    - atr: Average True Range (Wilder's smoothing)
    - sma: Simple moving average of close
    - close: close prices as numpy array
    - high: high prices as numpy array
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame with columns close, high, low")

    for col in ("close", "high", "low"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain column '{col}'")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)

    # MACD: difference of EMAs
    fast_ema = close_s.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR (True Range) - Wilder's smoothing
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Fill first TR if NaN
    if np.isnan(tr.iat[0]):
        tr.iat[0] = high_s.iat[0] - low_s.iat[0]
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    return {
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
        "close": close_s.values,
        "high": high_s.values,
    }


def order_func(*args: Any) -> Any:
    """
    Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    Expected calling signature from vectorbt: order_func(order_ctx, *order_args)
    where order_args are the arrays we passed (close, high, macd, signal, atr, sma, trailing_mult).

    The function creates and returns a vectorbt Order (or NoOrder) for the current context.

    Strategy:
    - Entry: MACD crosses above Signal AND price > SMA -> target 100% (TargetPercent)
    - Exit: MACD crosses below Signal OR price < highest_since_entry - trailing_mult * ATR -> target 0%

    The function uses OrderContext.pos_record_now['entry_idx'] to find entry index and compute
    highest_since_entry from the high price array between entry_idx and current index.
    """
    # Parse args
    if len(args) == 0:
        raise ValueError("order_func requires at least the order context and indicator arrays")

    # First arg is expected to be OrderContext
    ctx = args[0]
    # Remaining args expected: close, high, macd, signal, atr, sma, trailing_mult
    if len(args) < 8:
        # Some versions of vbt might pass arrays differently; try to locate first array-like arg
        arr_start = None
        for idx, a in enumerate(args):
            if isinstance(a, (np.ndarray, pd.Series, list)):
                arr_start = idx
                break
        if arr_start is None:
            raise ValueError("Could not find indicator arrays in order_func arguments")
        # Adjust parsing
        close = np.asarray(args[arr_start + 0], dtype=float)
        high = np.asarray(args[arr_start + 1], dtype=float)
        macd = np.asarray(args[arr_start + 2], dtype=float)
        signal = np.asarray(args[arr_start + 3], dtype=float)
        atr = np.asarray(args[arr_start + 4], dtype=float)
        sma = np.asarray(args[arr_start + 5], dtype=float)
        trailing_mult = float(args[arr_start + 6])
    else:
        close = np.asarray(args[1], dtype=float)
        high = np.asarray(args[2], dtype=float)
        macd = np.asarray(args[3], dtype=float)
        signal = np.asarray(args[4], dtype=float)
        atr = np.asarray(args[5], dtype=float)
        sma = np.asarray(args[6], dtype=float)
        trailing_mult = float(args[7])

    # Import vectorbt enums and types
    import vectorbt as vbt

    SizeType = vbt.portfolio.enums.SizeType
    Direction = vbt.portfolio.enums.Direction
    Order = vbt.portfolio.enums.Order
    NoOrder = vbt.portfolio.NoOrder

    # Helper to get element or slice respecting 1D/2D arrays
    def get_elem(arr: np.ndarray, i: int, col: int = 0):
        if arr is None:
            return np.nan
        if arr.ndim == 1:
            return arr[i]
        # arr.ndim >= 2
        return arr[i, col]

    def get_range_max(arr: np.ndarray, start: int, end: int, col: int = 0):
        # return nan if invalid range
        if start is None or start < 0 or start > end:
            # fallback to current bar only
            return get_elem(arr, end, col)
        if arr.ndim == 1:
            return np.nanmax(arr[start : end + 1])
        return np.nanmax(arr[start : end + 1, col])

    # Context indices
    try:
        i = int(getattr(ctx, "i"))
    except Exception:
        # Fallback to index attribute if present
        i = int(getattr(ctx, "index", 0))
    try:
        col = int(getattr(ctx, "col", 0))
    except Exception:
        col = 0

    # Guard bounds
    n = close.shape[0]
    if i < 0 or i >= n:
        return NoOrder

    # Current and previous indicator values (use safe getters)
    price = get_elem(close, i, col)
    sma_curr = get_elem(sma, i, col) if sma is not None else np.nan
    atr_curr = get_elem(atr, i, col) if atr is not None else np.nan

    macd_curr = get_elem(macd, i, col)
    signal_curr = get_elem(signal, i, col)
    macd_prev = get_elem(macd, i - 1, col) if i > 0 else np.nan
    signal_prev = get_elem(signal, i - 1, col) if i > 0 else np.nan

    # Determine position state using context
    # position_now is the current position (number of shares). For long-only we check >0
    position_now = float(getattr(ctx, "position_now", 0.0))

    in_position = position_now != 0.0

    macd_cross_up = False
    macd_cross_down = False
    if not (np.isnan(macd_prev) or np.isnan(signal_prev) or np.isnan(macd_curr) or np.isnan(signal_curr)):
        macd_cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        macd_cross_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    # ENTRY: only if not in position
    if not in_position:
        if macd_cross_up and (not np.isnan(sma_curr)) and (price > sma_curr):
            # Target 100% of portfolio
            return Order(
                size=1.0,
                price=np.inf,
                size_type=int(SizeType.TargetPercent),
                direction=int(Direction.LongOnly),
                fees=0.0,
                fixed_fees=0.0,
                slippage=0.0,
            )
        else:
            return NoOrder

    # If in position, consider exits
    # Try to obtain entry index from pos_record_now (dtype trade_dt has entry_idx field)
    entry_idx = None
    try:
        pos_rec = getattr(ctx, "pos_record_now", None)
        if pos_rec is not None:
            # numpy void record access
            if isinstance(pos_rec, (np.void,)):
                # field may be present
                if "entry_idx" in pos_rec.dtype.names:
                    entry_idx = int(pos_rec["entry_idx"])
            else:
                # Fallback: try dictionary-like access
                try:
                    entry_idx = int(pos_rec.get("entry_idx", -1))
                except Exception:
                    entry_idx = None
    except Exception:
        entry_idx = None

    # Compute highest high since entry (inclusive)
    if entry_idx is None or entry_idx < 0:
        # Fall back to high from start
        start_idx = 0
    else:
        start_idx = entry_idx

    highest_since_entry = get_range_max(high, start_idx, i, col)

    # Trailing stop level
    trailing_ok = not np.isnan(atr_curr) and np.isfinite(highest_since_entry)
    trailing_level = highest_since_entry - trailing_mult * atr_curr if trailing_ok else -np.inf

    exit_on_trailing = trailing_ok and (price < trailing_level)
    exit_on_macd = macd_cross_down

    if exit_on_macd or exit_on_trailing:
        # Target 0% to close position
        return Order(
            size=0.0,
            price=np.inf,
            size_type=int(SizeType.TargetPercent),
            direction=int(Direction.LongOnly),
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )

    # Otherwise, no order
    return NoOrder
