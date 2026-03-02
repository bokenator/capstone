import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def _ensure_1d_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.ravel()


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    a_arr = _ensure_1d_array(close_a)
    b_arr = _ensure_1d_array(close_b)

    if len(a_arr) != len(b_arr):
        raise ValueError("close_a and close_b must have the same length")

    a = pd.Series(a_arr)
    b = pd.Series(b_arr)

    cov_ab = a.rolling(window=hedge_lookback, min_periods=2).cov(b)
    var_b = b.rolling(window=hedge_lookback, min_periods=2).var()

    hedge_ratio = cov_ab / var_b
    hedge_ratio = hedge_ratio.replace([np.inf, -np.inf], np.nan)
    hedge_ratio = hedge_ratio.fillna(method='ffill').fillna(0.0)

    spread = a - hedge_ratio * b

    spread_mean = spread.rolling(window=zscore_lookback, min_periods=2).mean()
    spread_std = spread.rolling(window=zscore_lookback, min_periods=2).std(ddof=0)
    spread_std = spread_std.replace(0.0, np.nan)

    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.fillna(0.0)

    return {
        "hedge_ratio": hedge_ratio.values.astype(float),
        "spread": spread.values.astype(float),
        "zscore": zscore.values.astype(float),
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    # Context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    position_now = float(getattr(c, "position_now", 0.0))

    close_a = _ensure_1d_array(close_a)
    close_b = _ensure_1d_array(close_b)
    zscore = _ensure_1d_array(zscore)
    hedge_ratio = _ensure_1d_array(hedge_ratio)

    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = float(zscore[i]) if not np.isnan(zscore[i]) else 0.0
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0

    if i > 0:
        z_prev = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else z
    else:
        z_prev = z

    exit_on_zero_cross = (z_prev * z) < 0
    exit_on_stop = abs(z) > float(stop_threshold)

    enter_short_a = z > float(entry_threshold)
    enter_long_a = z < -float(entry_threshold)

    if exit_on_zero_cross or exit_on_stop:
        pos_a_target = 0.0
        pos_b_target = 0.0
    else:
        if enter_short_a:
            if price_a <= 0 or np.isnan(price_a):
                return (np.nan, 0, 0)
            units_a = float(notional_per_leg) / price_a
            pos_a_target = -units_a
            pos_b_target = -hr * pos_a_target
        elif enter_long_a:
            if price_a <= 0 or np.isnan(price_a):
                return (np.nan, 0, 0)
            units_a = float(notional_per_leg) / price_a
            pos_a_target = units_a
            pos_b_target = -hr * pos_a_target
        else:
            return (np.nan, 0, 0)

    target = pos_a_target if col == 0 else pos_b_target

    # compute change needed
    order_change = target - position_now
    if np.isclose(order_change, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Use SizeType.Size == 1 and Direction.Net == 1 as safe defaults
    size_type_code = 1
    direction_code = 1

    # Return signed order change as size (allow negative to indicate sell in net mode)
    return (float(order_change), int(size_type_code), int(direction_code))
