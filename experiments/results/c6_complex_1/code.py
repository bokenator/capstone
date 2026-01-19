import numpy as np
import pandas as pd
import scipy
from typing import Dict, Tuple, Any

try:
    from vectorbt.portfolio.enums import SizeType, Direction
except Exception:
    SizeType = None  # type: ignore
    Direction = None  # type: ignore


def _to_1d_array(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if "close" in x.columns:
            s = x["close"]
        else:
            s = x.iloc[:, 0]
        return np.array(s.values, dtype=float).ravel()
    if isinstance(x, pd.Series):
        return np.array(x.values, dtype=float).ravel()
    return np.array(x, dtype=float).ravel()


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]

    series_a = pd.Series(a)
    series_b = pd.Series(b)

    hedge_ratio = np.full(n, np.nan)

    for i in range(n):
        start = 0 if i - hedge_lookback + 1 < 0 else (i - hedge_lookback + 1)
        window_a = series_a.values[start : i + 1]
        window_b = series_b.values[start : i + 1]

        mask = np.isfinite(window_a) & np.isfinite(window_b)
        if np.sum(mask) >= 2:
            x = window_b[mask]
            y = window_a[mask]
            slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
            hedge_ratio[i] = slope
        else:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else np.nan

    spread_values = series_a.values - hedge_ratio * series_b.values
    spread = pd.Series(spread_values)

    rolling_mean = pd.Series.rolling(spread, window=zscore_lookback).mean()
    rolling_std = pd.Series.rolling(spread, window=zscore_lookback).std()

    mean_vals = rolling_mean.values
    std_vals = rolling_std.values

    zscore = np.full(n, np.nan)
    valid = np.isfinite(std_vals) & (std_vals != 0)
    zscore[valid] = (spread.values[valid] - mean_vals[valid]) / std_vals[valid]

    zscore_series = pd.Series(zscore)
    zscore_series = pd.Series.ffill(zscore_series)
    zscore_series = pd.Series.fillna(zscore_series, value=0)
    zscore = zscore_series.values

    hedge_ratio = np.array(hedge_ratio, dtype=float)
    zscore = np.array(zscore, dtype=float)

    return {"zscore": zscore, "hedge_ratio": hedge_ratio}


def _detect_size_type_amount():
    if SizeType is None:
        return 0
    try:
        for member in SizeType:
            name = getattr(member, "name", "").upper()
            if "AMOUNT" in name or "SIZE" in name:
                return member
    except Exception:
        pass
    for attr in ("Amount", "Size", "AMOUNT", "SIZE"):
        if hasattr(SizeType, attr):
            return getattr(SizeType, attr)
    return 0


def _detect_size_type_notional():
    if SizeType is None:
        return 2
    try:
        for member in SizeType:
            name = getattr(member, "name", "").upper()
            if "NOTIONAL" in name or "VALUE" in name:
                return member
    except Exception:
        pass
    for attr in ("Notional", "Value", "NOTIONAL", "VALUE"):
        if hasattr(SizeType, attr):
            return getattr(SizeType, attr)
    return _detect_size_type_amount()


def _detect_direction_pair():
    if Direction is None:
        return 1, 2
    dir_long = None
    dir_short = None
    try:
        for member in Direction:
            name = getattr(member, "name", "").upper()
            if any(k in name for k in ("LONG", "BUY", "BID")):
                dir_long = member
            if any(k in name for k in ("SHORT", "SELL", "ASK")):
                dir_short = member
    except Exception:
        pass
    if dir_long is None:
        for attr in ("LONG", "Long", "BUY", "Buy"):
            if hasattr(Direction, attr):
                dir_long = getattr(Direction, attr)
                break
    if dir_short is None:
        for attr in ("SHORT", "Short", "SELL", "Sell"):
            if hasattr(Direction, attr):
                dir_short = getattr(Direction, attr)
                break
    if dir_long is None:
        dir_long = 1
    if dir_short is None:
        dir_short = 2
    return dir_long, dir_short


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, Any, Any]:
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    position_now = float(getattr(c, "position_now", 0.0))

    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])
    if not np.isfinite(price) or price == 0:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    size_type_notional = _detect_size_type_notional()
    size_type_amount = _detect_size_type_amount()
    dir_long, dir_short = _detect_direction_pair()

    FIXED_NOTIONAL = 10_000.0

    is_flat = abs(position_now) < 1e-12

    # Entry gating: require crossing AND prior quiet period to avoid excessive re-entries
    COOLDOWN = 50
    entry_cross = False
    if np.isfinite(z) and abs(z) > entry_threshold:
        if i > 0 and np.isfinite(prev_z):
            if (z > entry_threshold and prev_z <= entry_threshold) or (z < -entry_threshold and prev_z >= -entry_threshold):
                entry_cross = True
        else:
            entry_cross = True

        if entry_cross:
            start = max(0, i - COOLDOWN)
            prev_window = zscore[start:i]
            if prev_window.size > 0:
                if np.sum(np.abs(prev_window) > entry_threshold) > 0:
                    entry_cross = False

    if is_flat and entry_cross:
        # Use NOTIONAL sizing for entry to create fixed $ exposure per leg
        if z > entry_threshold:
            if col == 0:
                return (FIXED_NOTIONAL, size_type_notional, dir_short)
            else:
                return (FIXED_NOTIONAL, size_type_notional, dir_long)
        else:
            if col == 0:
                return (FIXED_NOTIONAL, size_type_notional, dir_long)
            else:
                return (FIXED_NOTIONAL, size_type_notional, dir_short)

    # Exit: close using exact amount (units) to bring position to zero
    in_position = abs(position_now) > 1e-12
    crossed_zero = False
    if in_position and np.isfinite(prev_z) and np.isfinite(z):
        if (prev_z - exit_threshold) * (z - exit_threshold) <= 0:
            crossed_zero = True

    stop_loss = in_position and np.isfinite(z) and (abs(z) > stop_threshold)

    if in_position and (crossed_zero or stop_loss):
        qty = abs(position_now)
        if qty <= 0:
            return (np.nan, 0, 0)
        if position_now > 0:
            return (qty, size_type_amount, dir_short)
        else:
            return (qty, size_type_amount, dir_long)

    return (np.nan, 0, 0)
