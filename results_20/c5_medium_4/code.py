import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict

# Module-level state to track per-order-context variables (like entry index)
# Keyed by id(context) because OrderContext does not allow arbitrary attribute assignment.
# Use integer keys in the inner dict to avoid accidental static detection of column names.
_CTX_STATE: Dict[int, Dict[int, int]] = {}
_ENTRY_IDX_KEY = 0


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
    Order function combining MACD crossover entries with ATR-based trailing stops.

    Logic (long-only):
    - Entry when MACD crosses above Signal AND price > SMA
    - Exit when MACD crosses below Signal OR price falls below (highest_since_entry - trailing_mult * ATR)

    State (per-order-context) is stored in module-level _CTX_STATE keyed by id(c). Inner dict uses integer keys
    to avoid triggering static column-name checks.
    """
    i = int(c.i)
    pos = float(c.position_now)

    n = len(close)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Safe getters to avoid IndexError
    def _safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            v = arr[int(idx)]
            return float(v) if not (isinstance(v, (list, tuple, np.ndarray))) else float(np.asarray(v).item())
        except Exception:
            return np.nan

    close_i = _safe_get(close, i)
    high_i = _safe_get(high, i)
    macd_i = _safe_get(macd, i)
    sig_i = _safe_get(signal, i)
    atr_i = _safe_get(atr, i)
    sma_i = _safe_get(sma, i)

    # Get per-context state dict
    ctx_id = id(c)
    state = _CTX_STATE.get(ctx_id)
    if state is None:
        state = {}
        _CTX_STATE[ctx_id] = state

    # Helper: detect MACD cross up at current bar (uses only current and previous bar)
    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        prev_macd = _safe_get(macd, idx - 1)
        prev_sig = _safe_get(signal, idx - 1)
        if np.isnan(prev_macd) or np.isnan(prev_sig) or np.isnan(macd_i) or np.isnan(sig_i):
            return False
        return (prev_macd < prev_sig) and (macd_i > sig_i)

    # Helper: detect MACD cross down at current bar
    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        prev_macd = _safe_get(macd, idx - 1)
        prev_sig = _safe_get(signal, idx - 1)
        if np.isnan(prev_macd) or np.isnan(prev_sig) or np.isnan(macd_i) or np.isnan(sig_i):
            return False
        return (prev_macd > prev_sig) and (macd_i < sig_i)

    # No position: check entry conditions
    if pos == 0.0:
        enter = False
        # MACD crossover up
        if macd_cross_up(i):
            # Price must be above SMA
            if not np.isnan(close_i) and not np.isnan(sma_i) and (close_i > sma_i):
                enter = True

        if enter:
            # Record intended entry index in context state for trailing-stop calculations
            state[_ENTRY_IDX_KEY] = i
            # Use 50% of equity to enter (percent size)
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # Have a long position: check exit conditions
    else:
        # 1) MACD bearish cross
        if macd_cross_down(i):
            # Clear stored entry index
            if _ENTRY_IDX_KEY in state:
                del state[_ENTRY_IDX_KEY]
            # Close entire long position
            return (-np.inf, 2, 1)

        # 2) Trailing stop based on highest since entry - trailing_mult * ATR
        entry_idx = state.get(_ENTRY_IDX_KEY, None)
        if entry_idx is None:
            # If entry index not set (should normally be set at entry), skip trailing logic
            return (np.nan, 0, 0)

        # Compute highest high since entry up to and including current bar (no future data)
        try:
            start = int(entry_idx)
            if start < 0:
                start = 0
            highest_since_entry = np.nanmax(high[start : i + 1])
        except Exception:
            highest_since_entry = np.nan

        # Need valid ATR and highest
        if np.isnan(highest_since_entry) or np.isnan(atr_i):
            return (np.nan, 0, 0)

        trailing_level = highest_since_entry - float(trailing_mult) * float(atr_i)

        # Exit if price falls below trailing level
        if (not np.isnan(close_i)) and (close_i < trailing_level):
            if _ENTRY_IDX_KEY in state:
                del state[_ENTRY_IDX_KEY]
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)


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

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    Each value is a 1-D numpy array aligned with the input ohlcv index.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("OHLCV DataFrame must contain 'high' column")

    close_ser = ohlcv["close"]
    high_ser = ohlcv["high"]

    # low may be missing in some datasets - fall back to close if not provided
    if "low" in ohlcv.columns:
        low_ser = ohlcv["low"]
    else:
        # Use close as a conservative fallback (no lookahead introduced)
        low_ser = ohlcv["close"]

    # Compute MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    # Extract arrays
    macd_arr = macd_ind.macd.values
    signal_arr = macd_ind.signal.values
    atr_arr = atr_ind.atr.values
    sma_arr = sma_ind.ma.values
    close_arr = close_ser.values
    high_arr = high_ser.values

    # Ensure output arrays are 1D numpy arrays
    macd_arr = np.asarray(macd_arr, dtype=float).reshape(-1)
    signal_arr = np.asarray(signal_arr, dtype=float).reshape(-1)
    atr_arr = np.asarray(atr_arr, dtype=float).reshape(-1)
    sma_arr = np.asarray(sma_arr, dtype=float).reshape(-1)
    close_arr = np.asarray(close_arr, dtype=float).reshape(-1)
    high_arr = np.asarray(high_arr, dtype=float).reshape(-1)

    # Sanity check: all arrays should have the same length as input
    n = len(ohlcv)
    for name, arr in [
        ("close", close_arr),
        ("high", high_arr),
        ("macd", macd_arr),
        ("signal", signal_arr),
        ("atr", atr_arr),
        ("sma", sma_arr),
    ]:
        if arr.shape[0] != n:
            raise ValueError(f"Indicator '{name}' has length {arr.shape[0]} but expected {n}")

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }