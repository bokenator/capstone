"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Rolling OLS hedge ratio (lookback)
- Z-score computed from spread (rolling mean/std)
- Entry/exit/stop logic implemented in order_func

This module intentionally avoids numba and returns simple Python tuples from order_func.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread.

    Args:
        close_a: Prices for asset A (1D numpy array).
        close_b: Prices for asset B (1D numpy array).
        hedge_lookback: Window length for rolling OLS (in bars).
        zscore_lookback: Window length for rolling mean/std of spread.

    Returns:
        A dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-scores for the spread (same length)

    Notes:
        - Uses scipy.stats.linregress on each rolling window of length hedge_lookback.
        - Warmup periods are filled with np.nan.
    """
    # Convert to numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = close_a.shape[0]

    # Initialize hedge ratio and spread
    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS hedge ratio: regress A ~ B for each window of exact length hedge_lookback
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        a_win = close_a[start : i + 1]
        b_win = close_b[start : i + 1]

        # Require enough valid observations
        mask = np.isfinite(a_win) & np.isfinite(b_win)
        if mask.sum() < 2:
            # keep hedge_ratio[i] as NaN
            continue

        try:
            # Regress A on B: A = alpha + beta * B + eps; take beta as hedge ratio
            slope, intercept, r_value, p_value, std_err = linregress(b_win[mask], a_win[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            # In case linregress fails for degenerate data
            hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    valid_hr = np.isfinite(hedge_ratio)
    for i in np.where(valid_hr)[0]:
        if np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
            spread[i] = close_a[i] - hedge_ratio[i] * close_b[i]

    # Compute rolling mean and std of spread for z-score
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to compute z-score
    rolling_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if (
            np.isfinite(spread[i])
            and np.isfinite(rolling_mean[i])
            and np.isfinite(rolling_std[i])
            and rolling_std[i] > 0
        ):
            zscore[i] = float((spread[i] - rolling_mean[i]) / rolling_std[i])

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


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
    Order function for flexible multi-asset pairs trading.

    This function is called once per column per bar (wrapped by the provided
    flexible wrapper in the backtest runner). It returns a simple tuple
    (size, size_type, direction) or a tuple whose first element is np.nan to
    indicate no order.

    SizeType mapping used:
      - 0: Amount (number of units)
      - 3: TargetAmount (set position amount to this value)

    Direction mapping used:
      - 0: LongOnly
      - 1: ShortOnly
      - 2: Both (used for TargetAmount close)

    Logic implemented:
      - Entry when zscore > entry_threshold: short A, long B
      - Entry when zscore < -entry_threshold: long A, short B
      - Exit when zscore crosses 0.0 or when |zscore| > stop_threshold (stop-loss)
      - Position sizing uses fixed notional_per_leg to compute approximate unit sizes
        for Asset A; Asset B size is derived from hedge_ratio to respect the hedge.

    Notes:
      - Returns (np.nan, 0, 0) when no order should be placed for this column.
      - For closing positions we use TargetAmount=0 (size_type=3, direction=2).

    Args:
        c: Context (provides attributes i (index), col (column index), position_now, cash_now).
        close_a: 1D numpy array of asset A close prices.
        close_b: 1D numpy array of asset B close prices.
        zscore: 1D numpy array of z-scores (same length as prices).
        hedge_ratio: 1D numpy array of hedge ratios (same length as prices).
        entry_threshold: Threshold to enter trades (e.g., 2.0).
        exit_threshold: Threshold for exit crossing (should be 0.0 here).
        stop_threshold: Stop-loss threshold on abs(zscore) (e.g., 3.0).
        notional_per_leg: Notional size per leg in dollars (e.g., 10000.0).

    Returns:
        Tuple[size (float), size_type (int), direction (int)].
    """
    # Default: no order
    NO_ORDER: Tuple[float, int, int] = (np.nan, 0, 0)

    # Read index and column
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Basic bounds check
    if i < 0 or i >= len(zscore):
        return NO_ORDER

    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If indicator unavailable, do nothing
    if not np.isfinite(z):
        return NO_ORDER

    # Stop-loss: if exceeded, close positions (target 0)
    if np.isfinite(z) and abs(z) > stop_threshold:
        return (0.0, 3, 2)  # TargetAmount=0, Both directions

    # Exit when z-score crosses zero: compare with previous value
    prev_z = zscore[i - 1] if i - 1 >= 0 else np.nan
    if np.isfinite(prev_z) and np.sign(prev_z) != np.sign(z):
        # Crossed zero (including sign change to/from 0)
        return (0.0, 3, 2)

    # Determine price for this column
    price_a = close_a[i]
    price_b = close_b[i]

    # Position now for this column (number of units or position indicator)
    pos_now = getattr(c, "position_now", 0.0)

    # Entry: z > entry_threshold -> short A, long B
    if z > entry_threshold:
        # If this column already has a position, do nothing
        if pos_now is not None and abs(pos_now) > 1e-12:
            return NO_ORDER

        # Compute base units for asset A using notional_per_leg
        if not (np.isfinite(price_a) and price_a > 0):
            return NO_ORDER

        base_units_a = int(np.floor(notional_per_leg / price_a))
        base_units_a = max(1, base_units_a)

        if col == 0:
            # Asset A: SHORT base_units_a (Amount)
            return (float(base_units_a), 0, 1)  # Amount, ShortOnly
        else:
            # Asset B: LONG hr * base_units_a (Amount), fallback to at least 1
            if not np.isfinite(hr) or hr == 0:
                b_units = max(1, int(base_units_a))
            else:
                b_units = max(1, int(round(abs(hr) * base_units_a)))

            return (float(b_units), 0, 0)  # Amount, LongOnly

    # Entry: z < -entry_threshold -> long A, short B
    if z < -entry_threshold:
        if pos_now is not None and abs(pos_now) > 1e-12:
            return NO_ORDER

        if not (np.isfinite(price_a) and price_a > 0):
            return NO_ORDER

        base_units_a = int(np.floor(notional_per_leg / price_a))
        base_units_a = max(1, base_units_a)

        if col == 0:
            # Asset A: LONG base_units_a
            return (float(base_units_a), 0, 0)  # Amount, LongOnly
        else:
            # Asset B: SHORT hr * base_units_a
            if not np.isfinite(hr) or hr == 0:
                b_units = max(1, int(base_units_a))
            else:
                b_units = max(1, int(round(abs(hr) * base_units_a)))

            return (float(b_units), 0, 1)  # Amount, ShortOnly

    # Otherwise, no order
    return NO_ORDER
