"""
Pairs Trading Strategy Implementation

Exports:
- compute_spread_indicators
- order_func

Implements rolling OLS hedge ratio, spread, rolling z-score, and a flexible order function
for vectorbt's flexible order API.

Notes:
- Hedge ratio is computed with a rolling window up to `hedge_lookback`. For early bars
  when fewer than hedge_lookback points are available, an OLS is computed on the
  available history (min 2 points). This avoids NaNs after short warmup periods.
- Z-score uses rolling mean/std with `zscore_lookback` and min_periods=1.
- Order function sizes legs using a fixed notional per leg for Asset A and scales
  Asset B by the hedge ratio (units_B = hedge_ratio * units_A). Cash-sized orders
  are constructed (size in USD) to avoid ambiguity in size types.

CRITICAL: This code does not use numba decorators. It uses vectorbt enums only to
obtain the integer codes for order creation when available; fallbacks are provided
so the code remains robust if enums are inaccessible.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Try to import vectorbt enums to obtain exact integer codes for orders.
# Fall back to conservative defaults if unavailable.
try:
    import vectorbt as vbt  # type: ignore
    from vectorbt.portfolio.enums import SizeType, Direction  # type: ignore

    SIZE_TYPE_CASH_INT = int(SizeType.CASH)
    DIRECTION_LONG_INT = int(Direction.LONG)
    DIRECTION_SHORT_INT = int(Direction.SHORT)
except Exception:
    # Fallback integer codes. These are reasonable defaults used by vectorbt in many versions.
    SIZE_TYPE_CASH_INT = 0
    DIRECTION_LONG_INT = 1
    DIRECTION_SHORT_INT = 2


def _to_1d_array(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input to 1D numpy float array."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def compute_spread_indicators(
    prices_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    prices_b: Optional[Union[np.ndarray, pd.Series]] = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (rolling OLS), spread, and z-score for a pair of assets.

    Args:
        prices_a: Price series for asset A. If a DataFrame is passed and prices_b is None,
            the first column is used as A and the second column as B.
        prices_b: Price series for asset B. Optional if prices_a is a 2-column DataFrame.
        hedge_lookback: Rolling window length for hedge ratio OLS. For early bars with fewer
            points, an OLS is computed on available history (min 2 points).
        zscore_lookback: Rolling window length for z-score mean/std.

    Returns:
        Dict with keys:
            - 'hedge_ratio': numpy array of hedge ratios (length n)
            - 'spread': numpy array of spreads (length n)
            - 'zscore': numpy array of z-scores (length n)
            - 'rolling_mean': rolling mean of spread
            - 'rolling_std': rolling std of spread
    """
    # Support calling with a single DataFrame
    if prices_b is None and isinstance(prices_a, pd.DataFrame):
        if prices_a.shape[1] < 2:
            raise ValueError("prices_a DataFrame must have at least two columns when prices_b is None")
        pa = prices_a.iloc[:, 0]
        pb = prices_a.iloc[:, 1]
    else:
        pa = prices_a  # type: ignore
        pb = prices_b  # type: ignore

    if pb is None:
        raise ValueError("prices_b must be provided if prices_a is not a 2-column DataFrame")

    a = _to_1d_array(pa)
    b = _to_1d_array(pb)

    if len(a) != len(b):
        raise ValueError("Input price series must have the same length")

    n = len(a)

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    rolling_mean = np.full(n, np.nan, dtype=float)
    rolling_std = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio. For early bars we use available history (min 2 points)
    for t in range(n):
        start = 0 if t - hedge_lookback + 1 < 0 else t - hedge_lookback + 1
        x_win = b[start : t + 1]
        y_win = a[start : t + 1]

        # Mask NaNs in the window
        mask = (~np.isnan(x_win)) & (~np.isnan(y_win))
        if mask.sum() >= 2:
            x_valid = x_win[mask]
            y_valid = y_win[mask]
            # If constant series, slope will be 0
            try:
                slope = float(linregress(x_valid, y_valid).slope)
            except Exception:
                # Numerical fallback
                if np.allclose(x_valid, x_valid[0]):
                    slope = 0.0
                else:
                    slope = float(np.polyfit(x_valid, y_valid, 1)[0])
            hedge_ratio[t] = slope
        else:
            hedge_ratio[t] = np.nan

        # Spread at time t (use hedge_ratio at t if available)
        if not np.isnan(hedge_ratio[t]) and not np.isnan(a[t]) and not np.isnan(b[t]):
            spread[t] = a[t] - hedge_ratio[t] * b[t]
        else:
            spread[t] = np.nan

    # Rolling mean and std for z-score (min_periods=1 to reduce NaNs)
    for t in range(n):
        start = 0 if t - zscore_lookback + 1 < 0 else t - zscore_lookback + 1
        window = spread[start : t + 1]
        mask = ~np.isnan(window)
        if mask.sum() >= 1:
            w = window[mask]
            m = float(np.mean(w))
            s = float(np.std(w, ddof=0))
            # Prevent division by zero
            if s == 0 or np.isnan(s):
                s = 1e-8
            rolling_mean[t] = m
            rolling_std[t] = s
            if not np.isnan(spread[t]):
                zscore[t] = (spread[t] - m) / s
            else:
                zscore[t] = np.nan
        else:
            rolling_mean[t] = np.nan
            rolling_std[t] = np.nan
            zscore[t] = np.nan

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
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
    """
    Order function compatible with vectorbt flexible order API.

    The function returns a tuple: (size, size_type_int, direction_int)
    - size: amount (in USD when using cash size type)
    - size_type_int: integer code for how `size` is interpreted (cash/size/percent). We use cash.
    - direction_int: integer code indicating long (buy) or short (sell).

    Behavior:
    - Entry when |zscore| > entry_threshold
        - z > entry_threshold: Short A, Long B
        - z < -entry_threshold: Long A, Short B
    - Exit when zscore crosses 0 or when |zscore| > stop_threshold (stop-loss)

    Notes:
    - We size Asset A using `notional_per_leg` (cash). Asset B is sized in units as
      hedge_ratio * units_A; we convert to cash to submit orders (size in USD).
    - The function places orders only when needed (entry or exit). Otherwise returns NoOrder.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    curr_z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else 0.0

    # Current position for this column (units). Might be 0 if no position.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)
    eps_pos = 1e-8

    # If zscore or hedge_ratio undefined, do nothing
    if np.isnan(curr_z) or np.isnan(hedge_ratio[i]):
        return (np.nan, 0, 0)

    # Compute derived sizes
    units_a = 0.0
    if price_a > 0 and not np.isnan(price_a):
        units_a = float(notional_per_leg) / price_a
    units_b = float(hedge_ratio[i]) * units_a
    size_a_cash = float(notional_per_leg)
    size_b_cash = abs(units_b) * price_b

    # Determine signals
    stop_loss = abs(curr_z) > float(stop_threshold)
    crossed_zero = (prev_z * curr_z < 0) or np.isclose(curr_z, 0.0)

    entry_signal: Optional[str] = None
    if curr_z > float(entry_threshold):
        entry_signal = "short_a_long_b"
    elif curr_z < -float(entry_threshold):
        entry_signal = "long_a_short_b"

    # Exit logic (close existing positions)
    if stop_loss or (crossed_zero and abs(pos_now) > eps_pos):
        # Close this column's position fully
        if abs(pos_now) <= eps_pos:
            return (np.nan, 0, 0)
        # Size in cash to close: abs(units) * price
        size_cash = abs(pos_now) * (price_a if col == 0 else price_b)
        # Direction to close: if currently long (pos_now>0), sell; if short (pos_now<0), buy
        if pos_now > 0:
            direction = DIRECTION_SHORT_INT
        else:
            direction = DIRECTION_LONG_INT
        return (float(size_cash), int(SIZE_TYPE_CASH_INT), int(direction))

    # Entry logic: only enter if currently flat on this leg
    if entry_signal is not None and abs(pos_now) <= eps_pos:
        if col == 0:
            # Asset A leg
            if entry_signal == "short_a_long_b":
                # Short A
                return (float(size_a_cash), int(SIZE_TYPE_CASH_INT), int(DIRECTION_SHORT_INT))
            else:
                # Long A
                return (float(size_a_cash), int(SIZE_TYPE_CASH_INT), int(DIRECTION_LONG_INT))
        else:
            # Asset B leg
            if size_b_cash <= 0 or np.isnan(size_b_cash):
                return (np.nan, 0, 0)
            if entry_signal == "short_a_long_b":
                # Long B
                return (float(size_b_cash), int(SIZE_TYPE_CASH_INT), int(DIRECTION_LONG_INT))
            else:
                # Short B
                return (float(size_b_cash), int(SIZE_TYPE_CASH_INT), int(DIRECTION_SHORT_INT))

    # Otherwise, no order
    return (np.nan, 0, 0)


# Expose only the required symbols
__all__ = ["compute_spread_indicators", "order_func"]
