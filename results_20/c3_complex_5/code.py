import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score between two assets.

    Parameters
    ----------
    close_a : array-like
        Prices of asset A
    close_b : array-like
        Prices of asset B
    hedge_lookback : int
        Lookback window for rolling OLS. To avoid NaNs during warmup, a smaller
        window (t+1) is used when t+1 < hedge_lookback.
    zscore_lookback : int
        Lookback window for rolling mean/std of spread. Similarly uses a
        smaller window during warmup.

    Returns
    -------
    dict
        Dictionary with keys 'zscore', 'hedge_ratio', and 'spread' mapping to
        numpy arrays of the same length as inputs. All values are computed
        using only current and past information (no lookahead).
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    hedge_ratio = np.empty(n, dtype=float)
    spread = np.empty(n, dtype=float)
    zscore = np.empty(n, dtype=float)

    # Initialize with a reasonable default
    prev_slope = 1.0

    for t in range(n):
        # Rolling OLS: regress A on B using only data up to and including t.
        start_h = max(0, t - hedge_lookback + 1)
        x = a[start_h : t + 1]
        y = b[start_h : t + 1]

        if x.size < 2:
            slope = prev_slope if prev_slope is not None else 1.0
        else:
            ym = y.mean()
            xm = x.mean()
            denom = ((y - ym) ** 2).sum()
            if denom <= 1e-12:
                slope = prev_slope if prev_slope is not None else 1.0
            else:
                slope = ((x - xm) * (y - ym)).sum() / denom

        if not np.isfinite(slope):
            slope = prev_slope if prev_slope is not None else 1.0

        hedge_ratio[t] = slope
        prev_slope = slope

        # Spread at time t (uses current hedge ratio)
        spread[t] = a[t] - slope * b[t]

        # Rolling z-score (mean/std) using only past up to t
        start_z = max(0, t - zscore_lookback + 1)
        w = spread[start_z : t + 1]
        mean_w = np.nanmean(w)
        std_w = np.nanstd(w, ddof=0)

        if not np.isfinite(std_w) or std_w < 1e-12:
            zscore[t] = 0.0
        else:
            zscore[t] = (spread[t] - mean_w) / std_w

    return {"zscore": zscore, "hedge_ratio": hedge_ratio, "spread": spread}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading in flexible multi-asset mode.

    This function is called once per asset (column) per bar. It must return a
    tuple (size, size_type, direction). When size is NaN, the wrapper will
    interpret it as NoOrder for that column.

    Conventions used here:
    - size: absolute number of units (float)
    - size_type: 0 => absolute size (number of units)
    - direction: 1 => buy (long), 2 => sell (short)

    Logic (per specification):
    - Entry when zscore > entry_threshold: short A, long B (B sized by hedge_ratio)
    - Entry when zscore < -entry_threshold: long A, short B
    - Exit when zscore crosses 0: close both legs
    - Stop-loss when |zscore| > stop_threshold: close both legs
    - Position sizing: fixed notional_per_leg for asset A; asset B units = hedge_ratio * units_A

    The function uses only data up to the current bar (c.i), and reads the
    current position from c.position_now. It is deterministic and avoids
    lookahead.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if (zscore is not None and len(zscore) > i) else np.nan
    hr = float(hedge_ratio[i]) if (hedge_ratio is not None and len(hedge_ratio) > i) else 1.0

    # Current position for this column (units). May be positive (long) or negative (short).
    pos = getattr(c, "position_now", 0.0)
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        pos = 0.0
    pos = float(pos)

    # Helper to emit an order that closes the current position for this column
    def close_order() -> Tuple[float, int, int]:
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        size = abs(pos)
        # closing a long -> sell (2), closing a short -> buy (1)
        direction = 2 if pos > 0 else 1
        return (float(size), 0, int(direction))

    # 1) Stop-loss has highest priority: close if stop threshold exceeded
    if np.isfinite(z) and abs(z) > stop_threshold:
        return close_order()

    # 2) Exit on crossing zero (mean reversion). Use previous bar to detect crossing.
    if i > 0 and np.isfinite(z) and np.isfinite(zscore[i - 1]):
        z_prev = float(zscore[i - 1])
        crossed_to_zero = (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold)
        if crossed_to_zero:
            return close_order()

    # 3) Entry: only enter if current position is flat for this column
    if abs(pos) < 1e-12 and np.isfinite(z) and np.isfinite(hr) and price_a > 0:
        units_a = notional_per_leg / price_a
        units_b = abs(hr) * units_a

        # Short A, Long B
        if z > entry_threshold:
            if col == 0:
                return (float(units_a), 0, 2)
            else:
                return (float(units_b), 0, 1)

        # Long A, Short B
        if z < -entry_threshold:
            if col == 0:
                return (float(units_a), 0, 1)
            else:
                return (float(units_b), 0, 2)

    # No order for this column
    return (np.nan, 0, 0)
