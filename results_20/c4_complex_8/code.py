import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        zscore: Z-score of spread array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B

    # Defensive: ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(close_a)
    if not (0 <= i < n):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if i < len(zscore) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # Warmup checks
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Price validity
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Current position for this asset
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Determine number of shares per leg based on fixed notional
    # shares_a: number of Asset A shares for notional_per_leg
    # shares_b: number of Asset B shares scaled by hedge ratio (as per example guidance)
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Previous zscore for exit crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and i - 1 < len(zscore) else np.nan

    # Determine target positions for both assets
    target_a: Any = None
    target_b: Any = None

    # Stop-loss: close both legs immediately
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Exit condition: z-score crosses zero OR absolute zscore within exit_threshold
        crossed_zero = False
        if not np.isnan(prev_z):
            # crossing zero if signs differ (including hitting exactly zero)
            crossed_zero = (prev_z * z) < 0 or (z == 0.0 and prev_z != 0.0) or (prev_z == 0.0 and z != 0.0)

        within_exit = abs(z) <= abs(exit_threshold)

        if crossed_zero or within_exit:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry logic
            if z > entry_threshold:
                # Short Asset A, Long Asset B
                target_a = -shares_a
                target_b = shares_b
            elif z < -entry_threshold:
                # Long Asset A, Short Asset B
                target_a = shares_a
                target_b = -shares_b
            else:
                # No trading signal
                return (np.nan, 0, 0)

    # Choose which asset we are ordering for
    desired = target_a if col == 0 else target_b

    # Safety: if desired is None -> no action
    if desired is None:
        return (np.nan, 0, 0)

    # Compute delta in shares to send as order (size_type=0 -> amount/shares)
    delta_shares = float(desired - pos_now)

    # If no change, do nothing
    if abs(delta_shares) < 1e-8:
        return (np.nan, 0, 0)

    # Return order: use amount (shares), allow both long and short (direction=0)
    return (delta_shares, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible: it accepts either DataFrames/Series with a 'close'
    column/values or raw numpy arrays (1D) containing close prices.

    Args:
        asset_a: DataFrame or 1D array with 'close' prices for Asset A
        asset_b: DataFrame or 1D array with 'close' prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """

    def _to_close_array(x: Any) -> np.ndarray:
        # Accept pd.DataFrame with 'close' column, pd.Series, or numpy array
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame input must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        else:
            # Assume array-like
            arr = np.asarray(x)

        if arr.ndim != 1:
            raise ValueError('Close price input must be 1D')

        return arr.astype(float)

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 1 or hedge_lookback > n:
        raise ValueError('hedge_lookback must be between 1 and the length of the data')
    if zscore_lookback < 1 or zscore_lookback > n:
        raise ValueError('zscore_lookback must be between 1 and the length of the data')

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: slope of regression y (A) ~ x (B)
    for i in range(hedge_lookback, n + 1):
        start = i - hedge_lookback
        end = i
        x = close_b[start:end]
        y = close_a[start:end]

        # If any NaNs in window, skip
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i - 1] = np.nan
            continue

        # Require variance in x
        if np.isclose(np.std(x, ddof=0), 0.0):
            hedge_ratio[i - 1] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i - 1] = float(slope)
        except Exception:
            hedge_ratio[i - 1] = np.nan

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
