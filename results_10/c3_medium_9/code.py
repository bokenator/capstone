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

    Returns a dict with numpy arrays for keys: macd, signal, atr, sma, close, high.

    All returned arrays have the same length as the input DataFrame. The EMA/ATR
    use exponential smoothing (no future data). SMA uses a rolling mean with
    min_periods=sma_period so that SMA becomes valid at index (sma_period-1).
    """
    # Helper to fetch column in a case-insensitive manner
    def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        lookup = {col.lower(): col for col in df.columns}
        if name.lower() in lookup:
            return df[lookup[name.lower()]]
        raise KeyError(f"Column '{name}' not found in DataFrame")

    close_s = _get_col(ohlcv, "close").astype(float)
    high_s = _get_col(ohlcv, "high").astype(float)
    low_s = _get_col(ohlcv, "low").astype(float)

    # MACD: EMA fast - EMA slow, signal is EMA of MACD
    # Use ewm with adjust=False for recursive (no future) computation
    fast_ema = close_s.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_s = fast_ema - slow_ema
    signal_s = macd_s.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma_s = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR: True Range then Wilder's EMA (using ewm with alpha=1/period)
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    return {
        "macd": macd_s.values,
        "signal": signal_s.values,
        "atr": atr_s.values,
        "sma": sma_s.values,
        "close": close_s.values,
        "high": high_s.values,
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
) -> Tuple[float, Any, Any]:
    """
    Order function for vectorbt.from_order_func.

    Logic:
    - Simulate the strategy deterministically up to the current bar index (c.i)
      using only data up to that index to avoid lookahead.
    - If an entry occurs at the current bar -> return a target-long order (full size).
    - If an exit occurs at the current bar -> return a target-zero order (close).
    - Otherwise return np.nan to indicate no order.

    The function returns a tuple (size, size_type, direction). We try to pick a
    "Target" size type when available so that size is interpreted as a target
    position (1.0 = fully long, 0.0 = flat). If not available we fall back to
    Amount; the runner/wrapper will handle the conversion.
    """
    # Import enums lazily to avoid issues if this file is imported without vbt
    try:
        from vectorbt.portfolio.enums import SizeType, Direction
    except Exception:
        # Provide lightweight fallback integers if vectorbt is not available at import
        SizeType = type("_ST", (), {"Amount": 0})
        Direction = type("_D", (), {"Both": 0, "Long": 1})

    # Find a SizeType that represents a target if available
    size_type_target = None
    for attr in dir(SizeType):
        if "Target" in attr:
            size_type_target = getattr(SizeType, attr)
            break
    if size_type_target is None:
        # Fallback to Amount (will act as an absolute amount)
        size_type_target = getattr(SizeType, "Amount", 0)

    size_type_amount = getattr(SizeType, "Amount", size_type_target)
    dir_long = getattr(Direction, "Long", 1)
    dir_both = getattr(Direction, "Both", 0)

    # Ensure numpy arrays
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    # Current index in the simulation
    i = int(getattr(c, "i", len(close) - 1))
    if i < 0:
        i = 0
    n = len(close)
    if i >= n:
        i = n - 1

    in_position = False
    highest = -np.inf
    entry_idx = None

    action_at_i = None  # 'enter' or 'exit' or None

    # Simulate from t=1..i (we need previous value to detect MACD crosses)
    for t in range(1, i + 1):
        prev_m = macd[t - 1]
        prev_s = signal[t - 1]
        cur_m = macd[t]
        cur_s = signal[t]

        # Validity checks for MACD/Signal values
        macd_valid = not (np.isnan(prev_m) or np.isnan(prev_s) or np.isnan(cur_m) or np.isnan(cur_s))
        cross_up = macd_valid and (prev_m <= prev_s) and (cur_m > cur_s)
        cross_down = macd_valid and (prev_m >= prev_s) and (cur_m < cur_s)

        price_above_sma = (not np.isnan(sma[t])) and (close[t] > sma[t])

        if not in_position:
            if cross_up and price_above_sma:
                # Enter position at t
                in_position = True
                entry_idx = t
                highest = high[t] if not np.isnan(high[t]) else -np.inf
                if t == i:
                    action_at_i = "enter"
        else:
            # Update highest since entry
            if not np.isnan(high[t]):
                highest = max(highest, high[t])

            # Trailing stop check
            ts_ok = not np.isnan(atr[t]) and (highest > -np.inf)
            ts_tripped = ts_ok and (close[t] < (highest - trailing_mult * atr[t]))

            if ts_tripped or cross_down:
                # Exit at t
                in_position = False
                entry_idx = None
                highest = -np.inf
                if t == i:
                    action_at_i = "exit"

    # Decide what to return based on action_at_i
    if action_at_i == "enter":
        # Target full long position (1.0)
        return (1.0, size_type_target, dir_long)
    if action_at_i == "exit":
        # Target flat (0.0)
        return (0.0, size_type_target, dir_both)

    # No action
    return (np.nan, size_type_amount, dir_both)
