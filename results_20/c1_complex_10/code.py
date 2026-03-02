import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple


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

    This function is written for flexible (multi-asset) mode. It is called
    for each asset (column) separately with a light-weight OrderContext-like
    object `c` that exposes: c.i (index), c.col (0 for Asset A, 1 for Asset B),
    c.position_now (current position in shares), c.cash_now (cash available).

    Strategy logic implemented:
      - Hedge ratio: precomputed array (rolling OLS)
      - Spread z-score: precomputed array
      - Entry: z > entry_threshold -> short A, long B (hedge_ratio units)
               z < -entry_threshold -> long A, short B (hedge_ratio units)
      - Exit: z crosses 0 (sign change) -> close both legs
      - Stop-loss: |z| > stop_threshold -> close both legs
      - Sizing: base shares for Asset A = notional_per_leg / price_a
                Asset B units = hedge_ratio * base_shares (maintain qty hedge)

    Returns:
        (size, size_type, direction)
        size: number of shares to trade (positive=buy, negative=sell)
        size_type: 0=Amount (shares)
        direction: 0=Both

    Notes:
      - Do not use numba. This is plain Python.
      - If no action required, return (np.nan, 0, 0)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validations
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if (hedge_ratio is not None and i < len(hedge_ratio)) else np.nan

    # If any critical value is NaN, skip
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base shares (for Asset A) and corresponding hedge units for Asset B
    base_shares_a = notional_per_leg / price_a
    hedge_shares_b = hr * base_shares_a  # maintain quantity hedge: B units = hr * A units

    # Default desired positions (in shares)
    desired_a = 0.0
    desired_b = 0.0

    # Stop-loss takes precedence
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Detect zero crossing (exit) using previous z
        z_prev = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan
        crossed_zero = False
        if np.isfinite(z_prev):
            # sign change implies crossing zero
            if (z_prev * z) < 0:
                crossed_zero = True
            else:
                # Also treat small absolute values around exit_threshold as cross
                if abs(z) <= abs(exit_threshold) and abs(z_prev) > abs(exit_threshold):
                    crossed_zero = True

        if crossed_zero:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry logic
            if z > entry_threshold:
                # Short Asset A, Long Asset B (hr units)
                desired_a = -base_shares_a
                desired_b = hedge_shares_b
            elif z < -entry_threshold:
                # Long Asset A, Short Asset B (hr units)
                desired_a = base_shares_a
                desired_b = -hedge_shares_b
            else:
                # No new signal -> keep current position (no action)
                # We return no action by not creating orders (np.nan)
                desired_a = pos_now if col == 0 else desired_a
                # For B we cannot see pos of the other leg here; we'll use position_now for B when called

    # Determine which asset we're generating order for
    if col == 0:
        desired = float(desired_a)
    else:
        desired = float(desired_b)

    # If we haven't set a desired for the other asset (and no signal), signal to do nothing
    # Compute delta between desired and current position
    delta = desired - pos_now

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-9:
        return (np.nan, 0, 0)

    # Return absolute share delta as amount (shares)
    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with 'close' column or raw numpy arrays / pandas Series
    representing close prices. Returns a dict with numpy arrays for:
      - 'close_a'
      - 'close_b'
      - 'hedge_ratio'
      - 'zscore'

    Hedge ratio is computed with rolling OLS (using past `hedge_lookback` values,
    excluding the current bar). Z-score uses rolling mean/std of the spread
    with window `zscore_lookback`.
    """
    # Extract close arrays from inputs which may be DataFrames, Series or np.ndarray
    def _extract_close(x: Any, name: str) -> np.ndarray:
        # If DataFrame, only use 'close' column (validation requirement)
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError(f"DataFrame for {name} must contain 'close' column")
            arr = x["close"].values.astype(float)
            return arr
        # If Series
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # If numpy array or list-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            # If 2D, try to interpret as single-column close prices
            if arr.shape[1] == 1:
                return arr[:, 0]
            raise ValueError(f"Unexpected 2D array shape for {name}; expected 1D or single-column 2D")
        return arr

    close_a = _extract_close(asset_a, "asset_a")
    close_b = _extract_close(asset_b, "asset_b")

    if len(close_a) != len(close_b):
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)
    if n == 0:
        raise ValueError("Price arrays are empty")

    # Parameter validation
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")
    if hedge_lookback > n:
        raise ValueError("hedge_lookback cannot be larger than number of bars")

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each index i, regress y = close_a[window] on x = close_b[window]
    # using the previous `hedge_lookback` points (i-hedge_lookback .. i-1)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Remove any pairs with NaNs inside the window
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough points to compute regression
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio aligned with current index
    spread = np.full(n, np.nan, dtype=float)
    # Only compute spread where hedge_ratio is finite
    mask_hr = np.isfinite(hedge_ratio)
    spread[mask_hr] = close_a[mask_hr] - hedge_ratio[mask_hr] * close_b[mask_hr]

    # Rolling mean and std for spread
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    # Replace infs with NaN
    zscore[~np.isfinite(zscore)] = np.nan

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
