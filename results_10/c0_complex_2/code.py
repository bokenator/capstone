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
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: 1D array of close prices for asset A
        close_b: 1D array of close prices for asset B
        hedge_lookback: lookback window for rolling OLS to estimate hedge ratio
        zscore_lookback: lookback window for rolling mean/std for z-score

    Returns:
        A dictionary containing at least:
            - "hedge_ratio": np.ndarray of hedge ratios (same length as inputs)
            - "zscore": np.ndarray of spread z-scores
            - "spread": np.ndarray of spreads
            - "spread_mean": rolling mean of spread
            - "spread_std": rolling std of spread

    Notes:
        - Uses OLS regression of A ~ B (with intercept) on a rolling window to get slope.
        - Returns NaN where insufficient data or invalid regression.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: slope of regression A ~ B (with intercept)
    # Use numpy.linalg.lstsq for numerical stability
    for i in range(n):
        if i < hedge_lookback - 1:
            # Not enough data yet
            continue
        start = i - hedge_lookback + 1
        a_win = a[start : i + 1]
        b_win = b[start : i + 1]

        # If any NaNs, skip
        if np.isnan(a_win).any() or np.isnan(b_win).any():
            hedge_ratio[i] = np.nan
            continue

        # If B has zero variance, slope is undefined
        if np.allclose(b_win, b_win[0]):
            # Set hedge ratio to NaN when B window has no variation
            hedge_ratio[i] = np.nan
            continue

        # Prepare design matrix [B, 1]
        X = np.vstack([b_win, np.ones_like(b_win)]).T
        try:
            coef, _, _, _ = np.linalg.lstsq(X, a_win, rcond=None)
            slope = float(coef[0])
            hedge_ratio[i] = slope
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread = A - hedge_ratio * B (where hedge_ratio is available)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = a[valid_hr] - hedge_ratio[valid_hr] * b[valid_hr]

    # Rolling mean and std of spread for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to match many technical definitions
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
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
    Order function for flexible multi-asset pairs trading.

    This function is called once per asset (column) per bar in flexible mode.
    The wrapper will call this for both assets and convert returned tuples into
    vectorbt orders.

    Returns a tuple: (size, size_type, direction)
      - size: positive float representing number of units to trade (or np.nan for no order)
      - size_type: integer code for size interpretation (we use 0 => "units")
      - direction: integer code for buy/sell (1 => long/buy, 2 => short/sell)

    Notes:
      - Uses notional_per_leg to size the base A leg: base_units_A = notional_per_leg / price_A
      - B units are set to hedge_ratio * base_units_A to respect the hedge ratio
      - Entry: zscore > entry_threshold => Short A, Long B
               zscore < -entry_threshold => Long A, Short B
      - Exit: zscore crosses 0 (sign change) or |zscore| > stop_threshold => close both legs

    Important: This function avoids numba and vectorbt enums per instructions. It returns
    plain Python types (floats and ints). The wrapper transforms them to vectorbt orders.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 -> asset_a, 1 -> asset_b

    # Defensive access to arrays
    a = np.asarray(close_a, dtype=float).flatten()
    b = np.asarray(close_b, dtype=float).flatten()

    n = len(a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = a[i]
    price_b = b[i]

    # Current position in units (from context) - may be 0 if not provided
    position_now = float(getattr(c, "position_now", 0.0))

    # Read indicators at current bar
    z = float(zscore[i]) if (i < len(zscore) and np.isfinite(zscore[i])) else np.nan
    h = float(hedge_ratio[i]) if (i < len(hedge_ratio) and np.isfinite(hedge_ratio[i])) else np.nan

    # Basic validation
    if not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)
    if not np.isfinite(z) or not np.isfinite(h):
        # No signal or hedge ratio not available -> do nothing
        return (np.nan, 0, 0)
    if not np.isfinite(notional_per_leg) or notional_per_leg <= 0:
        return (np.nan, 0, 0)

    # Determine if we should close positions due to stop-loss
    close_all = False
    if abs(z) > stop_threshold:
        close_all = True

    # Determine if z-score crossed zero since last bar (requires previous zscore)
    crossed_zero = False
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = zscore[i - 1]
        if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
            crossed_zero = True

    # Compute desired positions (in units) for both assets from the perspective of this bar
    desired_a: float | None = None
    desired_b: float | None = None

    # Close on stop or zero-cross
    if close_all or crossed_zero:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Entry logic
        # Base units for A are sized by notional on A's side
        # base_units_A > 0 always
        if price_a == 0:
            return (np.nan, 0, 0)
        base_units_a = notional_per_leg / price_a

        if z > entry_threshold:
            # Short A, Long B
            desired_a = -base_units_a
            desired_b = float(h * base_units_a)
        elif z < -entry_threshold:
            # Long A, Short B
            desired_a = base_units_a
            desired_b = float(-h * base_units_a)
        else:
            # No signal and not closing -> do nothing
            return (np.nan, 0, 0)

    # Select target for this column
    target = desired_a if col == 0 else desired_b
    # Current position is given per-column
    delta = target - position_now

    # If delta is effectively zero, skip order
    if abs(delta) < 1e-12:
        return (np.nan, 0, 0)

    # Size in units to trade
    size = float(abs(delta))

    # Size type: 0 -> units (absolute number of shares)
    size_type = 0

    # Direction: 1 -> buy (increase position), 2 -> sell (decrease position)
    # When delta > 0 => we need to buy; when delta < 0 => sell
    direction = 1 if delta > 0 else 2

    return (size, int(size_type), int(direction))
