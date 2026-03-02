import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


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

    Returns a dictionary with numpy arrays for the following keys:
    - 'close', 'high' : price arrays
    - 'macd', 'signal' : MACD line and signal line
    - 'atr' : Average True Range
    - 'sma' : Simple Moving Average (period = sma_period)

    All arrays have the same length as ohlcv and contain np.nan for warmup values.
    """
    # Basic validation
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    # Ensure numeric dtype
    close = ohlcv["close"].astype(float).copy()
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()

    # MACD (EMA-based)
    # Use adjust=False for typical trading indicator behavior
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    # Use min_periods=sma_period so SMA is NaN until full window is available
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR (Average True Range)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Use EWM for ATR smoothing (common choice). This will produce NaN for initial bars.
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_line.values,
        "signal": signal_line.values,
        "atr": atr.values,
        "sma": sma.values,
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
    Order function implementing:
    - Entry: MACD crosses above signal AND price > 50-period SMA
    - Exit: MACD crosses below signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction).
    - size: positive to buy, negative to sell; np.nan to indicate no order
    - size_type: int (vectorbt enum value expected by the runner). We return plain ints here.
    - direction: int (vectorbt enum value expected by the runner).

    Notes:
    - This function does not import vectorbt enums (runner will handle conversion).
    - We try to be defensive when reading the execution context `c` since different vectorbt
      versions expose slightly different attributes on the position object.
    """
    # Helper to safely access array values
    def _val(arr: np.ndarray, idx: int) -> float:
        try:
            v = arr[idx]
        except Exception:
            return float("nan")
        # Convert numpy scalar to native float and preserve NaN
        try:
            return float(v)
        except Exception:
            return float("nan")

    # Current bar index
    i = int(getattr(c, "i", 0))

    # Minimal warmup: need at least one previous bar for MACD crossover checks
    if i <= 0:
        return (float("nan"), 0, 0)

    macd_prev = _val(macd, i - 1)
    signal_prev = _val(signal, i - 1)
    macd_curr = _val(macd, i)
    signal_curr = _val(signal, i)
    price = _val(close, i)
    sma_curr = _val(sma, i)

    # If indicators are not ready, do nothing
    if (
        np.isnan(macd_prev)
        or np.isnan(signal_prev)
        or np.isnan(macd_curr)
        or np.isnan(signal_curr)
        or np.isnan(price)
        or np.isnan(sma_curr)
    ):
        return (float("nan"), 0, 0)

    # Detect MACD crosses
    macd_cross_above = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
    macd_cross_below = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    # Determine whether we are currently in a (long) position.
    in_position = False
    entry_idx = None

    # The context `c` may expose .position with attributes; be defensive
    if hasattr(c, "position"):
        try:
            pos = getattr(c, "position")
            # Try several common attributes
            size_attr = getattr(pos, "size", None)
            if size_attr is None:
                size_attr = getattr(pos, "position_size", None)
            if size_attr is None:
                # Some versions provide `size` as a numpy array or scalar
                size_attr = getattr(pos, "_size", None)
            if size_attr is not None:
                try:
                    in_position = float(size_attr) != 0.0
                except Exception:
                    # If it's an array-like
                    try:
                        in_position = bool(size_attr)
                    except Exception:
                        in_position = False

            # If the position object has an explicit is_open flag, prefer it
            is_open_attr = getattr(pos, "is_open", None)
            if is_open_attr is not None:
                try:
                    in_position = bool(is_open_attr)
                except Exception:
                    pass

            # Try to get entry index
            entry_idx = getattr(pos, "entry_idx", None)
            if entry_idx is None:
                entry_idx = getattr(pos, "entry_i", None)
            if entry_idx is None:
                entry_idx = getattr(pos, "entry_index", None)
        except Exception:
            in_position = False

    else:
        # Fallback: some contexts expose position size directly on c
        pos_size = getattr(c, "position_size", None)
        if pos_size is not None:
            try:
                in_position = float(pos_size) != 0.0
            except Exception:
                in_position = False

    # ENTRY: MACD cross above AND price > SMA (trend filter)
    if (not in_position) and macd_cross_above and (price > sma_curr):
        # size_type=0 (Amount), direction=1 (Long) -- runner will interpret these ints
        return (1.0, 0, 1)

    # EXIT: if in position, check MACD cross down or ATR-based trailing stop
    if in_position:
        # Get ATR for current bar
        atr_curr = _val(atr, i)

        # Determine highest price since entry. If we don't have entry_idx fall back to start
        try:
            if entry_idx is None:
                start_idx = 0
            else:
                start_idx = int(entry_idx)
                if start_idx < 0:
                    start_idx = 0
        except Exception:
            start_idx = 0

        # Compute the highest high since entry (inclusive)
        try:
            if start_idx <= i:
                hi_since_entry = float(np.nanmax(high[start_idx : i + 1]))
            else:
                hi_since_entry = float(np.nanmax(high[: i + 1]))
        except Exception:
            hi_since_entry = float("nan")

        trail_price = float("nan")
        if (not np.isnan(hi_since_entry)) and (not np.isnan(atr_curr)):
            trail_price = hi_since_entry - float(trailing_mult) * float(atr_curr)

        # Exit on MACD cross down
        if macd_cross_below:
            return (-1.0, 0, 1)

        # Exit on trailing stop hit
        if (not np.isnan(trail_price)) and (price < trail_price):
            return (-1.0, 0, 1)

    # Default: no action
    return (float("nan"), 0, 0)
