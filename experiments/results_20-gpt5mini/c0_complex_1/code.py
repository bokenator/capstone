"""
Pairs trading strategy helpers for vectorbt backtests.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (lookback default 60)
- Spread = A - beta * B
- Z-score computed with rolling mean/std (lookback default 20)
- Entry: z > entry_threshold -> short A, long B; z < -entry_threshold -> long A, short B
- Exit: z crosses 0 OR |z| > stop_threshold -> close
- Position sizing: fixed notional per leg (value-based orders)

CRITICAL: This implementation does NOT use numba.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (1D array)
        close_b: Prices for asset B (1D array)
        hedge_lookback: Lookback window for rolling OLS (in bars)
        zscore_lookback: Lookback window for rolling mean/std of spread

    Returns:
        Dict with keys:
            - "hedge_ratio": np.ndarray of rolling hedge ratios (slope beta)
            - "spread": np.ndarray of spread values (A - beta * B)
            - "zscore": np.ndarray of spread z-scores

    Notes:
        - Uses full-window rolling regression (requires exactly `hedge_lookback` points)
        - If the denominator in OLS is numerically zero or there are NaNs in the window,
          the hedge_ratio and spread are set to NaN for that index.
        - Rolling mean/std for z-score require a full `zscore_lookback` of non-NaN
          spread observations (min_periods = zscore_lookback).
    """
    # Validate inputs and convert to numpy arrays
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = a.shape[0]

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Small epsilon for numerical stability
    eps = 1e-12

    # Rolling OLS: regress A on B for each full window of length hedge_lookback
    if hedge_lookback <= 0:
        raise ValueError("hedge_lookback must be > 0")

    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        win_a = a[start_idx : end_idx + 1]
        win_b = b[start_idx : end_idx + 1]

        # Skip windows with NaNs
        if np.isnan(win_a).any() or np.isnan(win_b).any():
            # hedge_ratio[end_idx] stays NaN
            continue

        mean_a = win_a.mean()
        mean_b = win_b.mean()

        # denom = sum((b - mean_b)**2)
        denom = np.sum((win_b - mean_b) ** 2)
        if denom <= eps:
            # Degenerate window; cannot compute slope
            continue

        cov_ab = np.sum((win_b - mean_b) * (win_a - mean_a))
        beta = cov_ab / denom
        hedge_ratio[end_idx] = beta

        # Compute spread at current bar
        spread[end_idx] = a[end_idx] - beta * b[end_idx]

    # Compute rolling mean and std of spread for z-score
    # Use pandas rolling to handle NaNs and min_periods requirement
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    # Use population std (ddof=0) to be consistent and avoid small-sample noise
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    zscore = (spread_series - roll_mean) / roll_std
    zscore = zscore.values.astype(float)

    # Clean up infinities
    zscore[np.isinf(zscore)] = np.nan

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
    }


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
    Order function for vectorbt in flexible multi-asset mode.

    The function returns a tuple (size, size_type, direction) where:
      - size: numeric size (we use value-based ordering in USD)
      - size_type: integer code for size type (we use 1 to indicate value-based orders)
      - direction: integer code for direction (1 = LONG / BUY, 2 = SHORT / SELL)

    Behavior:
      - Entry: when zscore > entry_threshold -> short A, long B
               when zscore < -entry_threshold -> long A, short B
               Orders are submitted only when the current position is flat (no exposure)
      - Exit: when zscore crosses zero OR |zscore| > stop_threshold -> close existing positions
      - Position sizing: fixed notional per leg (notional_per_leg), submitted as value orders

    Notes:
      - This function avoids numba and returns plain Python tuples
      - It expects `c` to provide attributes `i` (current index) and `col` (column index)
        and `position_now` (current position in units). The test harness provides a
        simulated context with these attributes.
    """
    # Configuration of returned enums (integers) - chosen to match vectorbt's typical
    # Value-based size type and direction encoding used by the from_order_func wrapper.
    # We use:
    #   size_type = 1  -> value-based orders (size interpreted in quote currency)
    #   direction = 1  -> LONG / BUY
    #   direction = 2  -> SHORT / SELL
    SIZE_TYPE_VALUE = 1
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    # Extract context information
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safe guards for indices
    if i < 0 or i >= len(zscore):
        return (float(np.nan), 0, 0)

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # Current z-score at this bar
    curr_z = zscore[i]
    if np.isnan(curr_z):
        return (float(np.nan), 0, 0)

    # Previous z-score for crossing detection
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Current position in units (can be positive/negative/zero)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Numerical tolerance for being considered flat
    tol = 1e-8

    # 1) Stop-loss: close if |z| > stop_threshold
    if not np.isnan(curr_z) and abs(curr_z) > float(stop_threshold):
        if abs(pos_now) <= tol:
            return (float(np.nan), 0, 0)
        # Close: compute value to close (units * price)
        close_value = abs(pos_now) * price
        # If currently long (pos_now > 0): to close we SELL (SHORT direction)
        direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
        return (float(close_value), SIZE_TYPE_VALUE, int(direction))

    # 2) Exit on zero crossing
    if i > 0 and (not np.isnan(prev_z)) and (prev_z * curr_z < 0):
        if abs(pos_now) <= tol:
            return (float(np.nan), 0, 0)
        close_value = abs(pos_now) * price
        direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
        return (float(close_value), SIZE_TYPE_VALUE, int(direction))

    # 3) Entry logic: only enter if currently flat
    if abs(pos_now) <= tol:
        # Short A / Long B when z > entry_threshold
        if curr_z > float(entry_threshold):
            if col == 0:
                # Short asset A
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)
            else:
                # Long asset B
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)

        # Long A / Short B when z < -entry_threshold
        if curr_z < -float(entry_threshold):
            if col == 0:
                # Long asset A
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_LONG)
            else:
                # Short asset B
                return (float(notional_per_leg), SIZE_TYPE_VALUE, DIRECTION_SHORT)

    # No order
    return (float(np.nan), 0, 0)
