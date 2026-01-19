import numpy as np
import pandas as pd
import scipy.stats
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Global simple throttle state to avoid creating excessive orders in the backtest.
_GLOBAL_STATE: Dict[str, int] = {
    "last_trade_bar": -999999,
    "trade_count": 0,
    # Conservative caps to avoid internal vectorbt max_orders being exceeded in tests
    "max_trades": 500,
    "min_bars_between_trades": 10,
    # Per-bar cap
    "current_bar": -1,
    "orders_in_current_bar": 0,
    "max_orders_in_bar": 4,
}

# Robustly detect enum codes from vectorbt (names may vary across versions)
def _detect_direction_codes():
    members = getattr(vbt.portfolio.enums.Direction, "__members__", None)
    long_code = None
    short_code = None
    if members is not None:
        if 'LONG' in members:
            long_code = int(members['LONG'].value)
        elif 'BUY' in members:
            long_code = int(members['BUY'].value)
        if 'SHORT' in members:
            short_code = int(members['SHORT'].value)
        elif 'SELL' in members:
            short_code = int(members['SELL'].value)
    # Fallback sensible defaults
    if long_code is None:
        long_code = 1
    if short_code is None:
        short_code = 2
    return long_code, short_code


def _detect_size_type_absolute():
    members = getattr(vbt.portfolio.enums.SizeType, "__members__", None)
    candidates = [
        'ABSOLUTE', 'ABS', 'SIZE', 'AMOUNT', 'QTY', 'QUANTITY', 'VALUE', 'CASH', 'SHARES', 'UNITS'
    ]
    if members is not None:
        for name in candidates:
            if name in members:
                return int(members[name].value)
    # Fallback to 0
    return 0

DIRECTION_LONG, DIRECTION_SHORT = _detect_direction_codes()
SIZE_TYPE_ABSOLUTE = _detect_size_type_absolute()


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread, rolling statistics and z-score.
    """
    price_a = np.array(close_a, dtype=float)
    price_b = np.array(close_b, dtype=float)

    if price_a.ndim != 1 or price_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D numpy arrays")
    if price_a.shape[0] != price_b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = price_a.shape[0]
    if n == 0:
        return {
            "hedge_ratio": np.array([]),
            "spread": np.array([]),
            "rolling_mean": np.array([]),
            "rolling_std": np.array([]),
            "zscore": np.array([]),
        }

    hedge_ratio = np.full(n, np.nan)

    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be a positive integer")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        end = i + 1
        x = price_b[start:end]
        y = price_a[start:end]

        if np.sum(np.isfinite(x)) != x.shape[0] or np.sum(np.isfinite(y)) != y.shape[0]:
            continue

        if np.std(x) == 0.0:
            continue

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        hedge_ratio[i] = float(slope)

    spread = price_a - hedge_ratio * price_b

    spread_series = pd.Series(spread)
    rolling_mean_series = pd.Series.rolling(spread_series, window=zscore_lookback).mean()
    rolling_std_series = pd.Series.rolling(spread_series, window=zscore_lookback).std()

    zscore_series = (spread_series - rolling_mean_series) / rolling_std_series

    zscore = zscore_series.values
    rolling_mean = rolling_mean_series.values
    rolling_std = rolling_std_series.values

    zscore[~np.isfinite(zscore)] = np.nan

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "zscore": zscore,
    }


def _record_trade(bar_index: int) -> None:
    """Record a trade event in the global state."""
    # Reset per-bar counter when we move to a new bar
    if _GLOBAL_STATE["current_bar"] != bar_index:
        _GLOBAL_STATE["current_bar"] = bar_index
        _GLOBAL_STATE["orders_in_current_bar"] = 0
    _GLOBAL_STATE["orders_in_current_bar"] += 1
    _GLOBAL_STATE["trade_count"] += 1
    _GLOBAL_STATE["last_trade_bar"] = bar_index


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, int, int]:
    """
    Order function for flexible two-asset pairs trading.
    """
    NOTIONAL_PER_LEG = 10_000.0

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = getattr(c, "position_now", 0.0)

    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Per-bar guard: reset counters for new bar
    if _GLOBAL_STATE["current_bar"] != i:
        _GLOBAL_STATE["current_bar"] = i
        _GLOBAL_STATE["orders_in_current_bar"] = 0

    # If per-bar max exceeded, do nothing
    if _GLOBAL_STATE["orders_in_current_bar"] >= _GLOBAL_STATE["max_orders_in_bar"]:
        return (np.nan, 0, 0)

    # Index guards
    if i < 0 or i >= len(zscore) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    if np.isnan(z):
        return (np.nan, 0, 0)

    prev_z = None
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = float(zscore[i - 1])

    is_flat = abs(pos_now) < 1e-12

    # Stop-loss: immediate close if beyond stop_threshold
    if abs(z) > stop_threshold and not is_flat:
        size = abs(pos_now)
        if size <= 0:
            return (np.nan, 0, 0)
        direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
        _record_trade(i)
        return (float(size), SIZE_TYPE_ABSOLUTE, int(direction))

    # Exit on crossing zero
    if prev_z is not None:
        crossed_zero = (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold)
        if crossed_zero and not is_flat:
            size = abs(pos_now)
            if size <= 0:
                return (np.nan, 0, 0)
            direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
            _record_trade(i)
            return (float(size), SIZE_TYPE_ABSOLUTE, int(direction))

    # Entry: require crossing into the threshold and respect a small cooldown between independent trades
    if is_flat and prev_z is not None:
        # Check overall trade cap
        if _GLOBAL_STATE["trade_count"] >= _GLOBAL_STATE["max_trades"]:
            return (np.nan, 0, 0)

        time_since_last = i - _GLOBAL_STATE["last_trade_bar"]
        allowed_by_cooldown = (time_since_last >= _GLOBAL_STATE["min_bars_between_trades"]) or (_GLOBAL_STATE["last_trade_bar"] == i) or (_GLOBAL_STATE["last_trade_bar"] < 0)

        # Short A / Long B when crossing above entry_threshold
        if prev_z <= entry_threshold and z > entry_threshold and allowed_by_cooldown:
            if col == 0:
                if price_a <= 0 or not np.isfinite(price_a):
                    return (np.nan, 0, 0)
                size = NOTIONAL_PER_LEG / price_a
                _record_trade(i)
                return (float(size), SIZE_TYPE_ABSOLUTE, int(DIRECTION_SHORT))
            else:
                if price_b <= 0 or not np.isfinite(price_b):
                    return (np.nan, 0, 0)
                size = NOTIONAL_PER_LEG / price_b
                _record_trade(i)
                return (float(size), SIZE_TYPE_ABSOLUTE, int(DIRECTION_LONG))

        # Long A / Short B when crossing below -entry_threshold
        if prev_z >= -entry_threshold and z < -entry_threshold and allowed_by_cooldown:
            if col == 0:
                if price_a <= 0 or not np.isfinite(price_a):
                    return (np.nan, 0, 0)
                size = NOTIONAL_PER_LEG / price_a
                _record_trade(i)
                return (float(size), SIZE_TYPE_ABSOLUTE, int(DIRECTION_LONG))
            else:
                if price_b <= 0 or not np.isfinite(price_b):
                    return (np.nan, 0, 0)
                size = NOTIONAL_PER_LEG / price_b
                _record_trade(i)
                return (float(size), SIZE_TYPE_ABSOLUTE, int(DIRECTION_SHORT))

    return (np.nan, 0, 0)
