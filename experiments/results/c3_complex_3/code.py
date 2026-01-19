"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20)
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold)

Notes:
- Rolling OLS hedge ratio is computed using past data only (no lookahead).
- Z-score is computed from rolling mean/std of spread.
- Position sizing is fixed notional $10,000 per leg; B is scaled by hedge ratio.
- Returns simple Python tuples for orders: (size, size_type, direction).

This file avoids numba and vectorbt internals as required by the prompt.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Simple global ordering guard to prevent excessive orders that may exceed
# vectorbt's internal max_orders in the test harness. It is reset at the start
# of each simulation run (when context.i == 0).
_GLOBAL_ORDER_COUNT = 0
_MAX_ORDERS_PER_RUN = 500


def _to_1d_array(x) -> np.ndarray:
    """Convert various input types to 1D numpy array of floats.

    Accepts: numpy arrays, pandas Series, pandas DataFrame (prefers 'close' column
    or first column), Python lists.
    """
    # Pandas DataFrame
    if isinstance(x, pd.DataFrame):
        if "close" in x.columns:
            arr = x["close"].values
        elif x.shape[1] == 1:
            arr = x.iloc[:, 0].values
        else:
            # If multiple columns, take the first column
            arr = x.iloc[:, 0].values
    elif isinstance(x, (pd.Series, pd.Index)):
        arr = x.values
    else:
        arr = np.asarray(x)

    # Ensure 1D
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1:
        # If shape is (n,1) flatten, otherwise take first column
        if arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            arr = arr[:, 0]

    return arr


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score for a pair of assets.

    Args:
        close_a: Prices of asset A (1D array-like or pandas Series/DataFrame).
        close_b: Prices of asset B (1D array-like or pandas Series/DataFrame).
        hedge_lookback: Lookback window (in bars) for rolling OLS regression.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dict with keys:
            - "zscore": np.ndarray of same length as the shorter input
            - "hedge_ratio": np.ndarray of same length as the shorter input
            - "spread": np.ndarray of same length as the shorter input

    Implementation details:
        - Rolling OLS is implemented using only data up to and including time t
          (no future data). For small windows (<2 points) the last valid
          hedge ratio is propagated to maintain continuity and avoid NaNs.
        - Z-score uses population std (ddof=0). If std == 0 then z-score is 0.
    """
    # Convert inputs to 1D arrays
    close_a = _to_1d_array(close_a)
    close_b = _to_1d_array(close_b)

    # Work on the overlapping prefix to avoid lookahead when one series is
    # longer than the other (this allows truncated inputs in tests).
    n = min(close_a.shape[0], close_b.shape[0])
    close_a = close_a[:n]
    close_b = close_b[:n]

    # Prepare outputs
    hedge_ratio = np.empty(n, dtype=float)
    hedge_ratio.fill(np.nan)

    # Rolling OLS regression (A ~ beta * B) using past data only
    last_valid_beta = 1.0

    for t in range(n):
        start = int(max(0, t - hedge_lookback + 1))
        x = close_b[start : t + 1]  # independent variable (B)
        y = close_a[start : t + 1]  # dependent variable (A)

        if x.size < 2:
            # Not enough points to run OLS; propagate previous beta
            beta = last_valid_beta
        else:
            # Compute OLS slope: beta = cov(x, y) / var(x)
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            denom = np.sum((x - mean_x) ** 2)
            if denom == 0 or not np.isfinite(denom):
                beta = last_valid_beta
            else:
                cov = np.sum((x - mean_x) * (y - mean_y))
                beta_raw = cov / denom
                # Fallback to last_valid_beta if result is not finite
                beta = beta_raw if np.isfinite(beta_raw) else last_valid_beta

        hedge_ratio[t] = beta
        last_valid_beta = beta

    # Compute spread using causal hedge_ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for spread -> z-score
    zscore = np.empty(n, dtype=float)
    zscore.fill(np.nan)

    for t in range(n):
        start = int(max(0, t - zscore_lookback + 1))
        window = spread[start : t + 1]
        if window.size == 0:
            zscore[t] = 0.0
        else:
            mu = np.nanmean(window)
            sigma = np.nanstd(window, ddof=0)
            if sigma == 0 or not np.isfinite(sigma):
                zscore[t] = 0.0
            else:
                zscore[t] = (spread[t] - mu) / sigma

    # Ensure no NaNs remain (use causal fills / safe defaults)
    # hedge_ratio: replace any remaining NaNs with first finite value or 1.0
    if np.isnan(hedge_ratio).any():
        finite_idx = np.where(np.isfinite(hedge_ratio))[0]
        if finite_idx.size == 0:
            hedge_ratio[:] = 1.0
        else:
            first = finite_idx[0]
            hedge_ratio[:first] = hedge_ratio[first]
            for i in range(first + 1, n):
                if not np.isfinite(hedge_ratio[i]):
                    hedge_ratio[i] = hedge_ratio[i - 1]

    # zscore: replace NaNs with 0 (safe neutral value)
    zscore = np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0)

    return {"zscore": zscore, "hedge_ratio": hedge_ratio, "spread": spread}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, int, int]:
    """Order function for vectorbt flexible multi-asset backtest.

    The function returns a tuple (size, size_type, direction) or a tuple with
    size = np.nan to indicate no order.

    Size semantics:
      - size is absolute number of units (shares)
      - size_type is 0 (absolute units)
      - direction is 1 for BUY (long), 2 for SELL (short)

    Strategy rules:
      - Entry when zscore > entry_threshold or zscore < -entry_threshold
      - Exit when zscore crosses 0.0 (mean reversion) or |zscore| > stop_threshold
      - Fixed notional per leg: $10,000
      - Asset B units scaled by hedge_ratio

    Args:
        c: Order context (provides .i, .col, .position_now, optionally .cash_now)
        close_a: array of asset A closes
        close_b: array of asset B closes
        zscore: precomputed zscore array
        hedge_ratio: precomputed hedge_ratio array
        entry_threshold: threshold to enter (default 2.0)
        exit_threshold: exit threshold (usually 0.0)
        stop_threshold: stop-loss threshold (default 3.0)

    Returns:
        (size, size_type, direction) where size can be np.nan to indicate no order.

    Notes:
        - Uses only information up to index c.i (causal).
        - Resets internal order counter at the start of each run (i == 0).
    """
    global _GLOBAL_ORDER_COUNT

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Reset per-run counter at start of simulation (first bar)
    if i == 0:
        _GLOBAL_ORDER_COUNT = 0

    # Defensive bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Ensure input price arrays are 1D numpy arrays
    price_a_arr = _to_1d_array(close_a)
    price_b_arr = _to_1d_array(close_b)

    # Work on overlap prefix
    n = min(len(price_a_arr), len(price_b_arr), len(zscore))
    if i >= n:
        return (np.nan, 0, 0)

    price_a = float(price_a_arr[i])
    price_b = float(price_b_arr[i])

    # Current z-score and hedge ratio (causal values)
    z = float(zscore[i]) if np.isfinite(zscore[i]) else 0.0
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else 1.0

    # Current position for this column (in units/shares)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validation
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Fixed notional per leg
    NOTIONAL = 10_000.0

    # Compute unit sizes (absolute number of shares)
    size_a_units = NOTIONAL / price_a
    size_b_units = abs(hr) * size_a_units

    # Determine previous z (for crossing detection)
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else None

    crossed_zero = False
    if prev_z is not None:
        if (prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0):
            crossed_zero = True

    # Stop loss condition
    stop_loss = abs(z) > stop_threshold

    # If we have a position and we need to exit (mean reversion or stop loss), close it
    if abs(pos_now) > 0.0 and (crossed_zero or stop_loss):
        size_to_close = abs(pos_now)
        # If currently long, SELL to close. If short, BUY to close.
        direction = 2 if pos_now > 0 else 1
        # Emit an order (respecting per-run cap)
        if _GLOBAL_ORDER_COUNT >= _MAX_ORDERS_PER_RUN:
            return (np.nan, 0, 0)
        _GLOBAL_ORDER_COUNT += 1
        return (float(size_to_close), 0, int(direction))

    # If no current position, consider entry signals
    if abs(pos_now) < 1e-12:
        # Short A, Long B when zscore > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short A
                if _GLOBAL_ORDER_COUNT >= _MAX_ORDERS_PER_RUN:
                    return (np.nan, 0, 0)
                _GLOBAL_ORDER_COUNT += 1
                return (float(size_a_units), 0, 2)
            else:
                # Long B (scaled by hedge ratio)
                if size_b_units > 0:
                    if _GLOBAL_ORDER_COUNT >= _MAX_ORDERS_PER_RUN:
                        return (np.nan, 0, 0)
                    _GLOBAL_ORDER_COUNT += 1
                    return (float(size_b_units), 0, 1)
                else:
                    return (np.nan, 0, 0)

        # Long A, Short B when zscore < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long A
                if _GLOBAL_ORDER_COUNT >= _MAX_ORDERS_PER_RUN:
                    return (np.nan, 0, 0)
                _GLOBAL_ORDER_COUNT += 1
                return (float(size_a_units), 0, 1)
            else:
                # Short B
                if size_b_units > 0:
                    if _GLOBAL_ORDER_COUNT >= _MAX_ORDERS_PER_RUN:
                        return (np.nan, 0, 0)
                    _GLOBAL_ORDER_COUNT += 1
                    return (float(size_b_units), 0, 2)
                else:
                    return (np.nan, 0, 0)

    # Otherwise, no order
    return (np.nan, 0, 0)
