import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


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
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If any critical value is NaN or non-positive, skip action
    if np.isnan(z) or np.isnan(hr) or not np.isfinite(price_a) or not np.isfinite(price_b):
        return (np.nan, 0, 0)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base shares for Asset A based on fixed notional per leg
    shares_a = float(notional_per_leg) / price_a
    # Apply hedge ratio to get Asset B shares (so that shares_b ~= hedge_ratio * shares_a)
    shares_b = shares_a * hr

    # Small tolerance to avoid churn
    tol = 1e-8

    # Previous z-score for crossing detection
    prev_z = np.nan
    if i > 0:
        prev_z_val = zscore[i - 1]
        if np.isfinite(prev_z_val):
            prev_z = float(prev_z_val)

    # Exit conditions: stop-loss or crossing exit_threshold (typically 0)
    crossed_exit = False
    if not np.isnan(prev_z):
        # Check for sign crossing or crossing the exit_threshold boundary
        if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
            crossed_exit = True
    stop_loss = abs(z) > stop_threshold

    if crossed_exit or stop_loss:
        # Close any existing position for this asset
        if abs(pos_now) <= tol:
            return (np.nan, 0, 0)
        # Return negative of current position to close
        return (-pos_now, 0, 0)

    # Entry logic
    # Short A, Long B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            target = -shares_a
        else:
            target = +shares_b
        delta = target - pos_now
        if abs(delta) <= tol:
            return (np.nan, 0, 0)
        return (delta, 0, 0)

    # Long A, Short B when z < -entry_threshold
    if z < -entry_threshold:
        if col == 0:
            target = +shares_a
        else:
            target = -shares_b
        delta = target - pos_now
        if abs(delta) <= tol:
            return (np.nan, 0, 0)
        return (delta, 0, 0)

    # No action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array/series)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array/series)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close prices from different input types
    def _extract_close(obj):
        # If DataFrame, use 'close' column
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame must contain 'close' column")
            s = obj['close'].astype(float).reset_index(drop=True)
            return s.values
        # If Series
        if isinstance(obj, pd.Series):
            return obj.astype(float).reset_index(drop=True).values
        # If numpy array
        if isinstance(obj, np.ndarray):
            return obj.astype(float)
        # Try to coerce
        try:
            return np.asarray(obj, dtype=float)
        except Exception:
            raise ValueError('Unsupported asset input type')

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Replace infinities with NaN for safety
    close_a = np.where(np.isfinite(close_a), close_a, np.nan)
    close_b = np.where(np.isfinite(close_b), close_b, np.nan)

    # Compute rolling hedge ratio using OLS on past data only (no lookahead).
    # We compute slope for each time i using data in [max(0, i-hedge_lookback+1) .. i].
    hedge_ratio = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]
        valid = (~np.isnan(x)) & (~np.isnan(y))
        if valid.sum() >= 2:
            try:
                slope, _, _, _, _ = stats.linregress(x[valid], y[valid])
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = 1.0
        else:
            # Fallback hedge ratio when insufficient data: neutral hedge of 1.0
            hedge_ratio[i] = 1.0

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score using past data only
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use population std (ddof=0) to avoid NaN when window size is 1
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Guard against zero std to avoid division by zero
    eps = 1e-8
    spread_std_safe = np.where(spread_std <= 0, eps, spread_std)

    zscore = (spread - spread_mean) / spread_std_safe

    # Ensure output arrays have same length and dtype
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }