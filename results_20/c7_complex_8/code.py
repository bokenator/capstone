import numpy as np
import pandas as pd
import vectorbt as vbt
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

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))  # Current position for this asset

    # Basic validations
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If z-score is NaN, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If prices invalid, skip
    if not (np.isfinite(price_a) and np.isfinite(price_b)) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio fallback
    hedge = float(hedge_ratio[i]) if (i < len(hedge_ratio) and np.isfinite(hedge_ratio[i])) else 1.0
    if hedge == 0 or not np.isfinite(hedge):
        hedge = 1.0

    # Determine share sizing. We size Asset A by fixed notional, Asset B scaled by hedge ratio
    shares_a = notional_per_leg / price_a
    shares_b = shares_a * hedge

    # Previous z-score for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan
    crossed_zero = False
    if i > 0 and np.isfinite(prev_z):
        crossed_zero = (prev_z * z) < 0

    # Priority: Stop-loss -> Exit on cross zero -> Entry
    target_a = None
    target_b = None

    if abs(z) > stop_threshold:
        # Stop-loss: close both legs
        target_a = 0.0
        target_b = 0.0
    elif crossed_zero:
        # Exit when spread reverts
        target_a = 0.0
        target_b = 0.0
    elif z > entry_threshold:
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

    # Choose target for this column
    target = target_a if col == 0 else target_b

    # If target is None (shouldn't happen), do nothing
    if target is None:
        return (np.nan, 0, 0)

    # If target is zero, close position if we have one
    if np.isclose(target, 0.0, atol=1e-12):
        if np.isclose(pos, 0.0, atol=1e-12):
            return (np.nan, 0, 0)
        # Return size to close entire position
        return (-pos, 0, 0)

    # Otherwise, compute delta to move to target
    delta = target - pos
    if np.isclose(delta, 0.0, atol=1e-12):
        return (np.nan, 0, 0)

    return (float(delta), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame or array-like with 'close' column/values for Asset A
        asset_b: DataFrame or array-like with 'close' column/values for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays

    Notes:
        - Uses a rolling OLS (stats.linregress) on past data only (no lookahead).
        - For early bars where full lookback isn't available, the regression uses all
          available past data (min 2 points). Hedge ratio is forward-filled if
          regression cannot be computed.
        - Rolling mean/std use pandas.rolling with min_periods=1 to avoid NaNs and
          ensure determinism. Std==0 results in z-score 0 to avoid infinities.
    """
    # Helper to extract close prices from various input types
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                return x['close'].to_numpy(dtype=float)
            # If DataFrame but no 'close' column, try to take first column
            if x.shape[1] >= 1:
                return x.iloc[:, 0].to_numpy(dtype=float)
            raise ValueError("asset DataFrame must contain a 'close' column")
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float)
        # Assume array-like
        arr = np.asarray(x, dtype=float)
        # If 2D and has named column, try to extract
        if arr.ndim == 2 and arr.shape[1] >= 1:
            return arr[:, 0]
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    # Align lengths if needed
    n = min(len(close_a), len(close_b))
    close_a = close_a[:n]
    close_b = close_b[:n]

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio (uses only past data up to current index)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]
        # Remove NaNs within the window
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x_valid = x[mask]
        y_valid = y[mask]
        if x_valid.size >= 2:
            # Compute slope (hedge ratio)
            try:
                slope = stats.linregress(x_valid, y_valid).slope
            except Exception:
                slope = np.nan
            if np.isfinite(slope):
                hedge_ratio[i] = float(slope)
            else:
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0
        else:
            # Not enough data: carry forward previous hedge ratio or default to 1.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0

    # Compute spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (use past data only)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use population std (ddof=0) to keep deterministic behavior for small windows
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Compute z-score, handling zero std
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Replace non-finite values (including division by zero) with 0.0
    bad_mask = ~np.isfinite(zscore) | (spread_std == 0)
    if np.any(bad_mask):
        zscore = zscore.copy()
        zscore[bad_mask] = 0.0

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
