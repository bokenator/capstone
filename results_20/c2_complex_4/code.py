"""
Pairs trading strategy helpers for vectorbt backtest runner.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> (size, size_type, direction)

Notes:
- No numba used.
- Uses rolling OLS for hedge ratio (analytic formulas via rolling sums).
- Orders are expressed in absolute amounts (SizeType.Amount) and use Direction.Both for flexibility.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# vectorbt enums are used to build order tuples (not using numba or nb helpers here)
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio (OLS), spread and z-score.

    Args:
        close_a: 1D array of asset A close prices.
        close_b: 1D array of asset B close prices.
        hedge_lookback: Lookback window (in bars) for rolling OLS to compute hedge ratio.
        zscore_lookback: Lookback window for spread mean/std used to compute z-score.

    Returns:
        Dict with keys:
            - "hedge_ratio": numpy array of rolling hedge ratios (same length as inputs)
            - "spread": numpy array of spread values (A - hedge_ratio * B)
            - "zscore": numpy array of z-score of the spread

    Notes:
        - Uses analytic OLS slope: slope = (sum(x*y) - sum(x)*sum(y)/n) / (sum(x^2) - sum(x)^2/n)
        - Rolling results are aligned to the right (value at index i uses window ending at i).
        - Returns NaN for indices without enough data to compute the respective rolling stat.
    """
    # Basic validation and conversions
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if a.shape[0] != b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")
    n_obs = a.shape[0]

    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    s_a = pd.Series(a)
    s_b = pd.Series(b)

    # Rolling sums for analytic OLS (regress A on B)
    win = int(hedge_lookback)
    # Use min_periods=win so we only compute when full window is available
    sum_x = s_b.rolling(window=win, min_periods=win).sum()
    sum_y = s_a.rolling(window=win, min_periods=win).sum()
    sum_xy = (s_b * s_a).rolling(window=win, min_periods=win).sum()
    sum_x2 = (s_b * s_b).rolling(window=win, min_periods=win).sum()

    denom = sum_x2 - (sum_x * sum_x) / win
    numer = sum_xy - (sum_x * sum_y) / win

    # Avoid division by zero
    small = 1e-12
    hedge_ratio = numer.copy()
    hedge_ratio.loc[denom.abs() > small] = (numer / denom).loc[denom.abs() > small]
    hedge_ratio.loc[denom.abs() <= small] = np.nan

    # Compute spread using the latest hedge ratio (aligned to the right)
    spread = s_a - hedge_ratio * s_b

    # Z-score using rolling mean and std of spread
    z_win = int(zscore_lookback)
    spread_mean = spread.rolling(window=z_win, min_periods=z_win).mean()
    # Use population std (ddof=0) to avoid NaNs when window is complete
    spread_std = spread.rolling(window=z_win, min_periods=z_win).std(ddof=0)

    zscore = (spread - spread_mean) / spread_std
    # Avoid division by zero / infinite z-scores
    zscore = zscore.where(spread_std > small)

    return {
        "hedge_ratio": hedge_ratio.to_numpy(dtype=float),
        "spread": spread.to_numpy(dtype=float),
        "zscore": zscore.to_numpy(dtype=float),
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
    """Order function used in flexible multi-asset mode.

    This function computes the desired target positions for both assets at the current bar
    and returns a single order for the asset indicated by c.col.

    Position sizing logic:
    - Base unit for Asset A = notional_per_leg / price_A
    - Asset A desired units: +/- base_unit (long if negative z-score, short if positive)
    - Asset B desired units: hedge_ratio * base_unit with sign opposite to Asset A to hedge

    Entry / Exit / Stop logic:
    - Entry: zscore > entry_threshold => short A, long B
             zscore < -entry_threshold => long A, short B
    - Exit: zscore crosses 0 (compared to previous bar) => close both legs
    - Stop-loss: |zscore| > stop_threshold => close both legs

    Returns:
        A tuple (size, size_type, direction)
        - size: absolute amount to trade (in units)
        - size_type: one of SizeType enum values (we use SizeType.Amount)
        - direction: one of Direction enum (we use Direction.Both to allow both buy/sell flows)

    If no order is needed, returns (np.nan, SizeType.Amount, Direction.Both) where size is NaN.
    """
    # Extract context information
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive casts
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    if i < 0 or i >= len(zscore):
        return (np.nan, int(SizeType.Amount), int(Direction.Both))

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z_val = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If indicators or prices are not available, do not place an order
    if any(np.isnan(x) for x in (price_a, price_b, z_val, hr)):
        return (np.nan, int(SizeType.Amount), int(Direction.Both))

    # Compute previous z-score for zero-cross detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Compute base unit for asset A (number of shares that correspond to notional_per_leg)
    if price_a <= 0 or not np.isfinite(price_a):
        return (np.nan, int(SizeType.Amount), int(Direction.Both))
    base_units_a = float(notional_per_leg) / price_a

    # Default: no change
    desired_a = 0.0
    desired_b = 0.0

    # Stop-loss: close both legs
    if abs(z_val) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Zero-cross exit (crossing through exit_threshold, typically 0.0)
        crossed_zero = False
        if not np.isnan(prev_z):
            if (prev_z > 0 and z_val <= exit_threshold) or (prev_z < 0 and z_val >= exit_threshold):
                crossed_zero = True

        if crossed_zero:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if z_val > entry_threshold:
                # Short A, Long B
                desired_a = -base_units_a
                desired_b = float(hr) * base_units_a
            elif z_val < -entry_threshold:
                # Long A, Short B
                desired_a = base_units_a
                desired_b = -float(hr) * base_units_a
            else:
                # No actionable signal -> no order
                return (np.nan, int(SizeType.Amount), int(Direction.Both))

    # Determine desired position for this column
    # c.position_now gives current held units for this column
    pos_now = float(getattr(c, "position_now", 0.0))
    desired_pos = desired_a if col == 0 else desired_b if col == 1 else 0.0

    # Compute delta
    delta = desired_pos - pos_now

    # Very small deltas are ignored
    if abs(delta) < 1e-8:
        return (np.nan, int(SizeType.Amount), int(Direction.Both))

    size = abs(delta)

    return (float(size), int(SizeType.Amount), int(Direction.Both))
