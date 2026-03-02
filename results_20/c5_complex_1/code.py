import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


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
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    # If index out of bounds, do nothing
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    # If zscore is not finite yet, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Prices must be positive and finite
    if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    h = hedge_ratio[i]
    if not np.isfinite(h):
        # Cannot size the hedge without a valid hedge ratio
        return (np.nan, 0, 0)

    # Compute desired share sizes (amount of units)
    # Base: choose Asset A shares so that notional_per_leg is allocated to asset A
    # Then scale Asset B by hedge ratio in unit terms: shares_b = hedge_ratio * shares_a
    shares_a = float(notional_per_leg / price_a)
    shares_b = float(h * shares_a)

    # Use small tolerance for zero comparisons
    tol = 1e-8
    in_position = not np.isclose(pos, 0.0, atol=tol)

    # STOP-LOSS: If |z| > stop_threshold then close any open positions
    if abs(z) > stop_threshold:
        if in_position:
            # Close current position for this asset
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # EXIT: If z-score crosses zero (sign change) -> close
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if np.isfinite(prev_z):
        crossed_zero = (prev_z * z) < 0

    if crossed_zero:
        if in_position:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Also allow exit when |z| <= exit_threshold (e.g., 0.0)
    if np.isfinite(exit_threshold) and abs(z) <= abs(exit_threshold):
        if in_position:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # ENTRY: if z-score breaches entry threshold, open pair
    # z > entry_threshold: short A, long B
    if z > entry_threshold:
        if col == 0:
            # Asset A: short
            # If no position, open short; if opposite sign, close it first
            if not in_position:
                return (-shares_a, 0, 0)
            # If currently long, close it
            if pos > 0:
                return (-pos, 0, 0)
            # Already short -> do nothing
            return (np.nan, 0, 0)
        else:
            # Asset B: long hedge_ratio units
            if not in_position:
                return (shares_b, 0, 0)
            if pos < 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # z < -entry_threshold: long A, short B
    if z < -entry_threshold:
        if col == 0:
            # Asset A: long
            if not in_position:
                return (shares_a, 0, 0)
            if pos < 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)
        else:
            # Asset B: short
            if not in_position:
                return (-shares_b, 0, 0)
            if pos > 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with a 'close' column, or 1D numpy arrays / pandas Series.

    Returns a dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays.
    """
    # Extract close price arrays from inputs (be flexible with input types)
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            arr = x['close'].to_numpy(dtype=float)
        elif isinstance(x, (pd.Series, np.ndarray, list, tuple)):
            arr = np.asarray(x, dtype=float)
        else:
            # Try to coerce
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError('Asset arrays must have the same shape')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 1:
        raise ValueError('hedge_lookback must be >= 1')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (using past data only up to current index i inclusive)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        end = i + 1  # inclusive of i

        x = close_b[start:end]
        y = close_a[start:end]

        # Require at least 2 valid observations to compute slope
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        # If x variance is zero, slope is undefined -> set slope to 0.0 to avoid blow-ups
        if np.isclose(np.nanvar(x_valid), 0.0):
            hedge_ratio[i] = 0.0
            continue

        try:
            slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
            # Ensure slope is finite
            if not np.isfinite(slope):
                hedge_ratio[i] = np.nan
            else:
                hedge_ratio[i] = float(slope)
        except Exception:
            # Fallback: use numpy polyfit
            try:
                p = np.polyfit(x_valid, y_valid, 1)
                slope = float(p[0])
                hedge_ratio[i] = slope
            except Exception:
                hedge_ratio[i] = np.nan

    # Compute spread using the hedge ratio at each time (no lookahead)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio) & np.isfinite(close_b) & np.isfinite(close_a)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score: use min_periods = zscore_lookback to match lookback
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Prevent division by zero by leaving zscore as NaN where std is 0 or NaN
    zscore = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[ok] = (spread[ok] - spread_mean[ok]) / spread_std[ok]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }