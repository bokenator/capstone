# Final version using vbt.order_nb to create order objects (fixed broadcasting)
from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """Compute indicators required by the strategy.

    Returns a dict with keys: close, high, macd, signal, atr, sma.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    for col in ["close", "high", "low"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv must contain '{col}' column")

    close_s = ohlcv["close"]
    high_s = ohlcv["high"]
    low_s = ohlcv["low"]

    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_s = macd_ind.macd
    signal_s = macd_ind.signal

    atr_s = vbt.ATR.run(high_s, low_s, close_s, window=atr_period).atr
    sma_s = vbt.MA.run(close_s, window=sma_period).ma

    # Forward-fill intermittent NaNs (uses only past data)
    macd_s = macd_s.fillna(method="ffill").fillna(method="bfill")
    signal_s = signal_s.fillna(method="ffill").fillna(method="bfill")
    atr_s = atr_s.fillna(method="ffill").fillna(method="bfill")
    sma_s = sma_s.fillna(method="ffill").fillna(method="bfill")

    return {
        "close": np.array(close_s.values, dtype=np.float64),
        "high": np.array(high_s.values, dtype=np.float64),
        "macd": np.array(macd_s.values, dtype=np.float64),
        "signal": np.array(signal_s.values, dtype=np.float64),
        "atr": np.array(atr_s.values, dtype=np.float64),
        "sma": np.array(sma_s.values, dtype=np.float64),
    }


def _resolve_enum_member(enum_obj, candidate_names, fallback_value=None):
    for name in candidate_names:
        if hasattr(enum_obj, name):
            return getattr(enum_obj, name)
    if fallback_value is not None:
        try:
            return enum_obj(fallback_value)
        except Exception:
            pass
    for attr in dir(enum_obj):
        if not attr.startswith("_"):
            try:
                val = getattr(enum_obj, attr)
                return val
            except Exception:
                continue
    raise AttributeError(f"Could not resolve enum member for candidates: {candidate_names}")


def order_func(
    price: Any,
    close: Any,
    high: Any,
    macd: Any,
    signal: Any,
    atr: Any,
    sma: Any,
    trailing_mult: float,
) -> np.ndarray:
    """Order function for vbt.Portfolio.from_order_func.

    Implements MACD crossover entries with ATR-based trailing stops.
    """
    close_arr = np.array(close, dtype=np.float64)
    high_arr = np.array(high, dtype=np.float64)
    macd_arr = np.array(macd, dtype=np.float64)
    signal_arr = np.array(signal, dtype=np.float64)
    atr_arr = np.array(atr, dtype=np.float64)
    sma_arr = np.array(sma, dtype=np.float64)

    n = len(close_arr)

    Direction = vbt.portfolio.enums.Direction
    SizeType = vbt.portfolio.enums.SizeType

    dir_long = _resolve_enum_member(Direction, ["Long", "LONG", "Buy", "BUY"], fallback_value=1)
    size_target_pct = _resolve_enum_member(SizeType, ["TargetPercent", "TARGET_PERCENT", "TargetPercent"], fallback_value=1)

    order_nb = vbt.portfolio.nb.order_nb

    # Create NoOrder template as order with price=inf (simulator checks isinf(order.price))
    no_order_template = order_nb(0.0, size_target_pct, 0, np.inf)

    # Initialize orders array by assigning per element to avoid broadcasting issues
    orders: np.ndarray = np.empty(n, dtype=object)
    for i in range(n):
        orders[i] = no_order_template

    in_position = False
    highest_since_entry = np.nan

    for i in range(n):
        if i == 0:
            continue

        macd_prev = macd_arr[i - 1]
        signal_prev = signal_arr[i - 1]
        macd_curr = macd_arr[i]
        signal_curr = signal_arr[i]

        close_curr = close_arr[i]
        high_curr = high_arr[i]
        atr_curr = atr_arr[i]
        sma_curr = sma_arr[i]

        macd_cross_up = (np.isfinite(macd_prev) and np.isfinite(signal_prev) and
                         np.isfinite(macd_curr) and np.isfinite(signal_curr) and
                         (macd_prev <= signal_prev) and (macd_curr > signal_curr))
        price_above_sma = (np.isfinite(close_curr) and np.isfinite(sma_curr) and (close_curr > sma_curr))

        if (not in_position) and macd_cross_up and price_above_sma:
            # Enter full long (target 100% allocation)
            ord_obj = order_nb(1.0, size_target_pct, dir_long, close_curr)
            orders[i] = ord_obj
            in_position = True
            highest_since_entry = high_curr if np.isfinite(high_curr) else np.nan
            continue

        if in_position:
            if np.isfinite(high_curr):
                if not np.isfinite(highest_since_entry):
                    highest_since_entry = high_curr
                else:
                    highest_since_entry = float(np.maximum(highest_since_entry, high_curr))

            trailing_trigger = False
            if np.isfinite(highest_since_entry) and np.isfinite(atr_curr):
                trail_level = highest_since_entry - (trailing_mult * atr_curr)
                if np.isfinite(trail_level) and np.isfinite(close_curr):
                    trailing_trigger = close_curr < trail_level

            macd_cross_down = (np.isfinite(macd_prev) and np.isfinite(signal_prev) and
                               np.isfinite(macd_curr) and np.isfinite(signal_curr) and
                               (macd_prev >= signal_prev) and (macd_curr < signal_curr))

            if trailing_trigger or macd_cross_down:
                ord_obj = order_nb(0.0, size_target_pct, dir_long, close_curr)
                orders[i] = ord_obj
                in_position = False
                highest_since_entry = np.nan

    return orders
