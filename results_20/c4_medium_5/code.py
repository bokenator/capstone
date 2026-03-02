import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Global state to track entry index and highest price since entry per column
# Stored as a list per column: [entry_index: Optional[int], highest_price: float]
_ENTRY_STATE: Dict[int, list] = {}


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
    Generate order at each bar. Called by vectorbt's from_order_func.

    Strategy:
    - Entry when MACD crosses above Signal AND price > SMA
    - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Position sizing:
    - Entry: use 50% of equity (size_type=2, size=0.5)
    - Exit: close entire position (size=-np.inf, size_type=2)

    This function keeps a small global state (_ENTRY_STATE) keyed by c.col (or 0 if not available)
    to remember the entry index and highest price since entry. The state is a list [entry_idx, highest].
    """
    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position size (0.0 if flat)

    # Column identifier (support multi-column though the backtest is single-asset)
    col = int(getattr(c, 'col', 0) if hasattr(c, 'col') else 0)

    # Initialize state for this column if not present
    st = _ENTRY_STATE.setdefault(col, [None, -np.inf])

    # Basic bounds check
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper functions
    def is_finite_val(x):
        return np.isfinite(x)

    def macd_cross_up(idx: int) -> bool:
        if idx <= 0:
            return False
        vals = [macd[idx - 1], signal[idx - 1], macd[idx], signal[idx]]
        if any(np.isnan(v) for v in vals):
            return False
        return (macd[idx - 1] < signal[idx - 1]) and (macd[idx] > signal[idx])

    def macd_cross_down(idx: int) -> bool:
        if idx <= 0:
            return False
        vals = [macd[idx - 1], signal[idx - 1], macd[idx], signal[idx]]
        if any(np.isnan(v) for v in vals):
            return False
        return (macd[idx - 1] > signal[idx - 1]) and (macd[idx] < signal[idx])

    # When flat, ensure stored state is cleared
    if pos == 0.0:
        st[0] = None
        st[1] = -np.inf

        # Entry conditions
        entry_cond = False
        try:
            price = float(close[i])
        except Exception:
            return (np.nan, 0, 0)

        # MACD cross up
        if macd_cross_up(i):
            # Price above SMA
            if not np.isnan(sma[i]) and is_finite_val(sma[i]):
                if price > float(sma[i]):
                    entry_cond = True

        if entry_cond:
            # Record entry index and initial highest price
            st[0] = i
            # Use high if available and finite, otherwise fallback to close
            hval = high[i] if (i < len(high) and is_finite_val(high[i])) else price
            st[1] = float(hval)

            # Enter with 50% of equity
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # If we are in a position
    # Ensure entry index is set: if not, set it to the current bar as a fallback
    if st[0] is None:
        st[0] = i
        st[1] = float(high[i]) if is_finite_val(high[i]) else float(close[i])

    # Update highest price since entry
    try:
        hcur = float(high[i]) if is_finite_val(high[i]) else float(close[i])
    except Exception:
        hcur = float(close[i])

    if not np.isnan(hcur) and is_finite_val(hcur):
        # Only update if current high is finite
        st[1] = max(st[1], hcur)

    # Exit conditions
    # 1) MACD cross down
    exit_macd = macd_cross_down(i)

    # 2) Price falls below trailing stop (highest_since_entry - trailing_mult * ATR)
    exit_trail = False
    highest = st[1]
    atr_val = atr[i] if (i < len(atr)) else np.nan
    if highest is not None and is_finite_val(highest) and is_finite_val(atr_val):
        trail_level = highest - trailing_mult * float(atr_val)
        # Use close price to evaluate
        if float(close[i]) < trail_level:
            exit_trail = True

    if exit_macd or exit_trail:
        # Reset state
        st[0] = None
        st[1] = -np.inf
        # Close entire long position
        return (-np.inf, 2, 1)

    # No action
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
    Precompute all indicators using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
    All values are numpy arrays of the same length as the input OHLCV DataFrame.
    """
    # Validate required columns
    if "close" not in ohlcv.columns:
        raise ValueError("`ohlcv` must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("`ohlcv` must contain 'high' column")

    close_series = ohlcv["close"].astype(float)
    high_series = ohlcv["high"].astype(float)

    # Use low if available, otherwise fall back to close (best-effort)
    if "low" in ohlcv.columns:
        low_series = ohlcv["low"].astype(float)
    else:
        low_series = close_series.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    # Ensure all arrays have the same length
    n = len(close_series)
    if not (len(high_series) == n == len(macd_arr) == len(signal_arr) == len(atr_arr) == len(sma_arr)):
        raise ValueError("Indicator arrays must all have the same length as input data")

    return {
        "close": close_series.values.astype(float),
        "high": high_series.values.astype(float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }
