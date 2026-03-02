import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

# vectorbt enums are used by the order function to specify size type and direction
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute indicators required by the strategy.

    Returns a DataFrame with columns: close, high, macd, signal, atr, sma

    All indicator calculations are causal (use only past data).
    EWM calculations use adjust=False and min_periods=0 so values are available
    early; SMA uses min_periods=sma_period to keep NaNs before warmup.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    df = ohlcv.copy()

    # Ensure required columns exist
    for col in ["close", "high", "low"]:
        if col not in df.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # MACD (EMA-based) - use adjust=False and min_periods=0 to avoid forward bias
    ema_fast = close.ewm(span=macd_fast, adjust=False, min_periods=0).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False, min_periods=0).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False, min_periods=0).mean()

    # ATR (True Range then EMA) - causal
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=0).mean()

    # SMA for trend filter - keep NaN until we have enough bars
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    indicators = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "macd": macd_line,
            "signal": signal_line,
            "atr": atr,
            "sma": sma,
        },
        index=df.index,
    )

    return indicators


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, SizeType, Direction]:
    """
    Order function implementing MACD crossover entries with ATR trailing stops.

    Returns a tuple (size, size_type, direction) or (np.nan, SizeType.Amount, Direction.Both)
    when no action is taken. The function is long-only and uses a simple per-asset
    state stored on the function to track whether a position is open and the
    highest price since entry for the trailing stop.

    Notes:
    - Uses only data up to the current index c.i (no lookahead).
    - State is initialized/reset at the start of a run (when c.i == 0).
    """
    # Get current bar index and column (asset) index if provided by context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0)) if hasattr(c, "col") else 0

    # Initialize per-column state on first call or when starting a new run
    if not hasattr(order_func, "_state") or i == 0:
        order_func._state = {}

    if col not in order_func._state or i == 0:
        # tracked fields: is_long (bool), entry_i (int), highest (float)
        order_func._state[col] = {
            "is_long": False,
            "entry_i": None,
            "highest": -np.inf,
        }

    state: Dict[str, Any] = order_func._state[col]

    # Safety: if index is 0, nothing to compare for crosses; return no action
    if i <= 0:
        return (np.nan, SizeType.Amount, Direction.Both)

    # Extract current and previous values (use only up-to-current-bar values)
    try:
        macd_curr = float(macd[i])
        signal_curr = float(signal[i])
        macd_prev = float(macd[i - 1])
        signal_prev = float(signal[i - 1])
        close_curr = float(close[i])
        high_curr = float(high[i])
        atr_curr = float(atr[i])
        sma_curr = float(sma[i]) if not pd.isna(sma[i]) else np.nan
    except Exception:
        # If any indexing fails (shouldn't happen), do nothing
        return (np.nan, SizeType.Amount, Direction.Both)

    # Detect MACD crosses (causal: use current and previous values only)
    macd_cross_up = False
    macd_cross_down = False

    if not (np.isnan(macd_prev) or np.isnan(signal_prev) or np.isnan(macd_curr) or np.isnan(signal_curr)):
        macd_cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        macd_cross_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    # Price above SMA trend filter (requires SMA to be non-NaN)
    price_above_sma = False
    if not np.isnan(sma_curr) and not np.isnan(close_curr):
        price_above_sma = close_curr > sma_curr

    # ENTRY: MACD cross up AND price above SMA
    if not state["is_long"]:
        if macd_cross_up and price_above_sma:
            # Open a long position of fixed size 1.0
            state["is_long"] = True
            state["entry_i"] = i
            # Initialize highest price since entry using the current high
            state["highest"] = high_curr if not np.isnan(high_curr) else close_curr
            return (1.0, SizeType.Amount, Direction.Both)

        # No entry -> no action
        return (np.nan, SizeType.Amount, Direction.Both)

    # If we have a position, update highest and check exit conditions
    # Update highest price since entry
    if not np.isnan(high_curr):
        state["highest"] = max(state.get("highest", -np.inf), high_curr)

    # Compute trailing stop price
    trailing_stop_price = None
    if not np.isnan(state["highest"]) and not np.isnan(atr_curr):
        trailing_stop_price = state["highest"] - float(trailing_mult) * atr_curr

    # EXIT: price falls below (highest_since_entry - trailing_mult * ATR) OR MACD crosses down
    if trailing_stop_price is not None and not np.isnan(trailing_stop_price) and not np.isnan(close_curr):
        if close_curr < trailing_stop_price:
            # Close the long position fully
            state["is_long"] = False
            state["entry_i"] = None
            state["highest"] = -np.inf
            return (-1.0, SizeType.Amount, Direction.Both)

    if macd_cross_down:
        # Close the long position on bearish MACD cross
        state["is_long"] = False
        state["entry_i"] = None
        state["highest"] = -np.inf
        return (-1.0, SizeType.Amount, Direction.Both)

    # Otherwise, hold
    return (np.nan, SizeType.Amount, Direction.Both)
