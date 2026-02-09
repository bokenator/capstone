import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict

# Global state store for order_func contexts (used because OrderContext may not allow arbitrary attributes)
# Store state as a small list to avoid use of string keys that static checks may flag
_ORDER_STATE: Dict[int, Any] = {}


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This implementation tracks per-context state in a global dictionary keyed by id(c),
    because the provided OrderContext does not allow adding arbitrary attributes.

    State layout (list):
      state[0] = highest price since entry (entry_high)
      state[1] = entered_at_idx
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Defensive indexing helper
    def safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return np.nan

    price = safe_get(close, i)
    high_i = safe_get(high, i)
    macd_i = safe_get(macd, i)
    signal_i = safe_get(signal, i)
    atr_i = safe_get(atr, i)
    sma_i = safe_get(sma, i)

    # Previous values for crossover detection
    macd_prev = safe_get(macd, i - 1) if i > 0 else np.nan
    signal_prev = safe_get(signal, i - 1) if i > 0 else np.nan

    # Get or create per-context state
    ctx_id = id(c)
    if ctx_id not in _ORDER_STATE:
        # Initialize as list: [entry_high, entered_idx]
        _ORDER_STATE[ctx_id] = [-np.inf, -1]
    state = _ORDER_STATE[ctx_id]

    # ENTRY: Only when flat (no position)
    if pos == 0.0:
        # Reset any previous entry tracking when flat
        state[0] = -np.inf
        state[1] = -1

        # MACD bullish crossover: current MACD > Signal AND previous MACD <= previous Signal
        macd_cross_up = (
            not np.isnan(macd_i) and not np.isnan(signal_i)
            and not np.isnan(macd_prev) and not np.isnan(signal_prev)
            and (macd_i > signal_i) and (macd_prev <= signal_prev)
        )

        # Trend filter: price above SMA
        above_sma = (not np.isnan(price)) and (not np.isnan(sma_i)) and (price > sma_i)

        if macd_cross_up and above_sma:
            # Record entry high as current high (if available) else use price
            state[0] = high_i if not np.isnan(high_i) else price
            state[1] = i

            # Enter long using 100% of equity
            # size_type=2 -> Percent, direction=1 -> LongOnly
            return (1.0, 2, 1)

        return (np.nan, 0, 0)

    # HAVE POSITION: Update highest price since entry and check exits
    else:
        # Update highest_since_entry
        if not np.isnan(high_i):
            state[0] = max(state[0], high_i)

        # Compute trailing stop: highest_since_entry - trailing_mult * ATR
        trailing_stop = np.nan
        if (state[0] != -np.inf) and (not np.isnan(atr_i)):
            trailing_stop = state[0] - float(trailing_mult) * atr_i

        # MACD bearish crossover on this bar
        macd_cross_down = (
            not np.isnan(macd_i) and not np.isnan(signal_i)
            and not np.isnan(macd_prev) and not np.isnan(signal_prev)
            and (macd_i < signal_i) and (macd_prev >= signal_prev)
        )

        # Price fell below trailing stop
        fell_below_trail = False
        if (not np.isnan(price)) and (not np.isnan(trailing_stop)):
            fell_below_trail = price < trailing_stop

        # Exit if any exit condition met
        if macd_cross_down or fell_below_trail:
            # Close entire position
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators. Use vectorbt indicator classes.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("Input ohlcv DataFrame must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("Input ohlcv DataFrame must contain 'high' column")

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)

    # Use 'low' if available, otherwise fall back to 'close' to avoid errors
    if "low" in ohlcv.columns:
        low = ohlcv["low"].astype(float)
    else:
        low = close.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close, window=sma_period)

    # Extract numpy arrays
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)
    close_arr = np.asarray(close.values, dtype=float)
    high_arr = np.asarray(high.values, dtype=float)

    # Ensure arrays are same length
    n = len(close_arr)
    if len(macd_arr) != n:
        macd_arr = np.resize(macd_arr, n)
    if len(signal_arr) != n:
        signal_arr = np.resize(signal_arr, n)
    if len(atr_arr) != n:
        atr_arr = np.resize(atr_arr, n)
    if len(sma_arr) != n:
        sma_arr = np.resize(sma_arr, n)
    if len(high_arr) != n:
        high_arr = np.resize(high_arr, n)

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }
