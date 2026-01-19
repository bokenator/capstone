import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy.stats
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    price_a = np.array(close_a, dtype=np.float64)
    price_b = np.array(close_b, dtype=np.float64)
    n = price_a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=np.float64)

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        y = price_a[start : i + 1]
        x = price_b[start : i + 1]
        if x.size == 0 or y.size == 0:
            hedge_ratio[i] = np.nan
            continue
        if (np.sum(np.isfinite(x)) != x.size) or (np.sum(np.isfinite(y)) != y.size):
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    spread = price_a - hedge_ratio * price_b
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    rolling_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values
    zscore = np.full(n, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(spread) & np.isfinite(rolling_mean) & np.isfinite(rolling_std) & (rolling_std != 0)
    zscore[valid_mask] = (spread[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]

    return {"hedge_ratio": hedge_ratio, "spread": spread, "zscore": zscore}


def _resolve_direction_values():
    D = vbt.portfolio.enums.Direction
    dir_members = []
    for name in dir(D):
        if name.startswith("_"):
            continue
        val = getattr(D, name)
        try:
            ival = int(val)
        except Exception:
            continue
        dir_members.append((name, ival))

    # Map known names
    dir_both = None
    dir_longonly = None
    dir_shortonly = None
    for name, ival in dir_members:
        uname = name.upper()
        if "BOTH" in uname:
            dir_both = getattr(D, name)
        if "LONG" in uname and "ONLY" in uname.upper():
            dir_longonly = getattr(D, name)
        if "SHORT" in uname and "ONLY" in uname.upper():
            dir_shortonly = getattr(D, name)
    # Fallbacks: try simpler matches
    if dir_longonly is None:
        for name, ival in dir_members:
            if "LONG" in name.upper():
                dir_longonly = getattr(D, name)
                break
    if dir_shortonly is None:
        for name, ival in dir_members:
            if "SHORT" in name.upper():
                dir_shortonly = getattr(D, name)
                break
    if dir_both is None:
        for name, ival in dir_members:
            if "BOTH" in name.upper() or "ANY" in name.upper() or "NET" in name.upper():
                dir_both = getattr(D, name)
                break

    # Final fallbacks to integers
    if dir_both is None:
        dir_both = 2
    if dir_longonly is None:
        dir_longonly = 0
    if dir_shortonly is None:
        dir_shortonly = 1

    return dir_both, dir_longonly, dir_shortonly


def _resolve_size_type_target_value():
    S = vbt.portfolio.enums.SizeType
    candidates = ["TargetValue", "TARGET_VALUE", "Value", "VALUE", "TARGETVALUE"]
    for name in candidates:
        if hasattr(S, name):
            return getattr(S, name)
    return 1


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, Any, Any]:
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    NO_ORDER = (np.nan, 0, 0)
    if i < 0 or i >= len(zscore):
        return NO_ORDER
    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if not np.isfinite(z) or not np.isfinite(hr):
        return NO_ORDER
    price_a = float(close_a[i])
    price_b = float(close_b[i])
    pos_now = float(getattr(c, "position_now", 0.0)) if getattr(c, "position_now", None) is not None else 0.0
    z_prev = zscore[i - 1] if i - 1 >= 0 else np.nan
    notional_per_unit_a = 10_000.0
    units_a = notional_per_unit_a / price_a
    units_b = hr * units_a
    target_value_b = units_b * price_b
    dir_both, dir_longonly, dir_shortonly = _resolve_direction_values()
    size_type = _resolve_size_type_target_value()
    # STOP-LOSS: if |z| > stop_threshold -> close any existing position
    if np.abs(z) > stop_threshold:
        if pos_now != 0.0:
            # Close by setting target value to 0 using Both direction
            return (0.0, size_type, dir_both)
        return NO_ORDER
    # EXIT: z-score crossed zero -> close existing positions
    if np.isfinite(z_prev) and (z_prev * z <= 0.0):
        if pos_now != 0.0:
            return (0.0, size_type, dir_both)
        return NO_ORDER
    # ENTRY: only enter when currently flat in this asset
    if pos_now == 0.0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Asset A: short
                return (notional_per_unit_a, size_type, dir_shortonly)
            else:
                # Asset B: long
                return (target_value_b, size_type, dir_longonly)
        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                return (notional_per_unit_a, size_type, dir_longonly)
            else:
                return (target_value_b, size_type, dir_shortonly)
    return NO_ORDER
