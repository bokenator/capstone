# Generated trading strategy combining MACD crossover entries with ATR-based trailing stops
# - compute_indicators: computes MACD, signal, ATR, SMA and returns numpy arrays
# - order_func: order function for vbt.Portfolio.from_order_func implementing entry/exit logic

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Global dictionary to hold persistent per-context state.
# We use id(c) as the key because OrderContext objects don't allow setting arbitrary attributes.
_CTX_STATE: Dict[int, Dict[str, Any]] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...]
        macd_fast: Fast EMA period for MACD
        macd_slow: Slow EMA period for MACD
        macd_signal: Signal EMA period for MACD
        sma_period: Period for trend filter SMA
        atr_period: Period for ATR

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        Values are numpy arrays with the same length as input.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain column '{col}'")

    # Work with float arrays to ensure numeric operations
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # MACD: difference of EMAs
    # Use ewm with adjust=False to avoid lookahead and produce values from the first bar
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA for trend filter (min_periods=1 to avoid NaNs early on)
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # True Range and ATR (use ewm for smoothing similar to Wilder's method but without lookahead)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use exponential smoothing for ATR to provide values early and avoid excessive NaNs
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    # Return numpy arrays for use inside the backtester
    return {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
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
    """Order function implementing MACD + ATR trailing stop strategy.

    The function returns a tuple (size, size_type, direction) which is then
    converted by the test harness into a proper vbt Order object. To keep
    the implementation independent of vectorbt enums, this function uses
    SizeType=0 (Amount) and Direction=0 (Both) and encodes buys as positive
    sizes and sells as negative sizes.

    State is stored in a module-level dictionary keyed by id(c). This avoids
    setting attributes on the OrderContext (which may not permit arbitrary
    attributes) while still keeping state local to the context/run.

    Args:
        c: vectorbt order function context (provides c.i)
        close, high, macd, signal, atr, sma: numpy arrays of indicators
        trailing_mult: multiplier for ATR to compute trailing stop (e.g., 2.0)

    Returns:
        (size, size_type, direction) tuple. size is float (positive to buy,
        negative to sell), size_type and direction are ints understood by
        the harness (we use 0 for both to represent Amount/Both).
    """
    # Safety checks
    i = int(getattr(c, "i", 0)) if hasattr(c, "i") else 0
    n = len(close)
    if i < 0 or i >= n:
        # Out of bounds, do nothing
        return (np.nan, 0, 0)

    # Use id(c) as a context key to store run-local state
    ctx_key = id(c)

    # Reinitialize state at the start of a run (when i == 0) or if not present
    if ctx_key not in _CTX_STATE or i == 0:
        _CTX_STATE[ctx_key] = {
            "in_pos": False,
            "entry_idx": -1,
            "entry_size": 0.0,
            "high_since_entry": -np.inf,
        }

    state = _CTX_STATE[ctx_key]

    # Read current bar values safely
    price = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan
    macd_i = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_i = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan

    # Determine MACD crosses using only current and previous bar (no lookahead)
    if i == 0:
        prev_macd = np.nan
        prev_signal = np.nan
    else:
        prev_macd = float(macd[i - 1]) if not np.isnan(macd[i - 1]) else np.nan
        prev_signal = float(signal[i - 1]) if not np.isnan(signal[i - 1]) else np.nan

    macd_cross_up = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and prev_macd <= prev_signal
        and macd_i > signal_i
    )

    macd_cross_down = (
        not np.isnan(prev_macd)
        and not np.isnan(prev_signal)
        and prev_macd >= prev_signal
        and macd_i < signal_i
    )

    # Long-only strategy
    # Use size_type=0 (Amount) and direction=0 (Both). Positive size to buy, negative to sell.
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_BOTH = 0

    # Entry logic
    if not state["in_pos"]:
        # Require valid SMA and price for robust behavior
        if macd_cross_up and (not np.isnan(sma_i)) and (not np.isnan(price)) and price > sma_i:
            # Open a long position with a fixed amount of 1.0 units
            size = 1.0
            state["in_pos"] = True
            state["entry_idx"] = i
            state["entry_size"] = size
            state["high_since_entry"] = high_i if not np.isnan(high_i) else price
            return (size, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

        # No entry
        return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # Position is open: update highest price since entry and check exits
    if not np.isnan(high_i):
        state["high_since_entry"] = max(state["high_since_entry"], high_i)

    # Compute trailing stop price; if ATR is not available, disable trailing stop
    if np.isnan(atr_i) or state["high_since_entry"] == -np.inf:
        trailing_stop_price = -np.inf
    else:
        trailing_stop_price = state["high_since_entry"] - trailing_mult * atr_i

    # Exit conditions: MACD bearish cross OR price drops below trailing stop
    if macd_cross_down or (not np.isnan(price) and price < trailing_stop_price):
        # Close the position by selling the same amount opened (negative size)
        close_size = -state["entry_size"] if state["entry_size"] != 0 else -1.0

        # Reset state
        state["in_pos"] = False
        state["entry_idx"] = -1
        state["entry_size"] = 0.0
        state["high_since_entry"] = -np.inf

        return (close_size, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)

    # No action this bar
    return (np.nan, SIZE_TYPE_AMOUNT, DIRECTION_BOTH)
