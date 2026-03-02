import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Tuple, Dict, Any


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
        c: vectorbt OrderContext-like with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A (numpy array)
        close_b: Close prices for Asset B (numpy array)
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

    Notes:
        - Uses delta sizing: returning (target_pos - current_pos, 0, 0) to adjust to target.
        - All computations use only data up to current index (no lookahead).
    """
    i: int = int(c.i)
    col: int = int(c.col)
    pos_now: float = float(getattr(c, 'position_now', 0.0))

    # Basic bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If z-score is not available, do nothing
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Avoid invalid prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio fallback
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else 1.0

    # Previous z-score for crossing detection
    z_prev = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Compute base share size for Asset A (fixed notional per leg)
    shares_a = float(notional_per_leg / price_a)

    # Determine target position for Asset A (in shares)
    pos_a_target: Any = None  # None means no new target (hold)

    # Stop-loss: if beyond stop threshold, close positions
    if abs(z) > stop_threshold:
        pos_a_target = 0.0
    else:
        # Exit on crossing of exit_threshold (e.g., crossing 0.0)
        if i > 0 and np.isfinite(z_prev):
            crossed = ((z_prev <= exit_threshold and z >= exit_threshold) or
                       (z_prev >= exit_threshold and z <= exit_threshold))
            # Only treat as crossing if signs changed or passed through
            if crossed:
                pos_a_target = 0.0

    # Entry logic (only if we don't already have a target from stop/exit)
    if pos_a_target is None:
        if z > entry_threshold:
            # Short Asset A
            pos_a_target = -shares_a
        elif z < -entry_threshold:
            # Long Asset A
            pos_a_target = shares_a

    # If still None, no action for this bar
    if pos_a_target is None:
        return (np.nan, 0, 0)

    # Compute corresponding target for Asset B so that pos_b = -hedge_ratio * pos_a
    pos_b_target = -hr * pos_a_target

    # Determine which asset we're creating order for and compute delta
    if col == 0:
        delta = float(pos_a_target - pos_now)
    else:
        delta = float(pos_b_target - pos_now)

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return order in shares (size_type=0), allow both long & short (direction=0)
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame or array with 'close' column/values for Asset A
        asset_b: DataFrame or array with 'close' column/values for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio (maximum window)
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Support both DataFrame/Series and numpy array inputs
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                return x['close'].astype(float).values
            # If the DataFrame is already a single-column of closes
            if x.shape[1] == 1:
                return x.iloc[:, 0].astype(float).values
            raise ValueError("asset DataFrame must contain a 'close' column")
        if isinstance(x, pd.Series):
            return x.astype(float).values
        if isinstance(x, np.ndarray):
            # 1D array
            if x.ndim == 1:
                return x.astype(float)
            # If 2D with single column
            if x.ndim == 2 and x.shape[1] == 1:
                return x[:, 0].astype(float)
            raise ValueError("Unsupported numpy array shape for close prices")
        # Fallback: try to convert
        try:
            s = pd.Series(x).astype(float)
            return s.values
        except Exception:
            raise TypeError("asset_a/asset_b must be array-like or DataFrame with 'close' column")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset arrays must have the same length")

    n = len(close_a)

    # Clean prices: replace infs and forward/backfill NaNs
    close_a = pd.Series(close_a).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    close_b = pd.Series(close_b).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS hedge ratio using only past data (exclude current index)
    # Use expanding window at the start to avoid NaNs (no lookahead)
    for i in range(n):
        start = max(0, i - hedge_lookback)
        end = i  # exclude current bar to avoid lookahead
        x = close_b[start:end]
        y = close_a[start:end]

        # Fallbacks for small windows
        if len(x) >= 2:
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
                hedge_ratio[i] = float(slope)
            except Exception:
                # If regression fails, fallback to previous value or 1.0
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and np.isfinite(hedge_ratio[i - 1]) else 1.0
        elif len(x) == 1:
            # Single point: use ratio
            hedge_ratio[i] = float(y[0] / x[0]) if x[0] != 0 else 1.0
        else:
            # No historical data: set to 1.0 as neutral hedge
            hedge_ratio[i] = 1.0

    # Spread using current prices and hedge ratio computed from past data
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for spread (use min_periods=1 so we have values early)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use population std (ddof=0) so early single-sample std is zero (we will handle zeros)
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score and guard against division by zero
    zscore = np.empty(n, dtype=float)
    eps = 1e-8
    for i in range(n):
        if not np.isfinite(spread[i]) or not np.isfinite(spread_mean[i]) or not np.isfinite(spread_std[i]):
            zscore[i] = np.nan
            continue
        if abs(spread_std[i]) < eps:
            # If std is effectively zero, set z-score to 0.0 (no deviation)
            zscore[i] = 0.0
        else:
            zscore[i] = float((spread[i] - spread_mean[i]) / spread_std[i])

    # Final pass: ensure no NaN after warmup by forward-filling any remaining NaNs
    # This preserves the property that values depend only on past data
    if np.isnan(zscore).any():
        # Forward-fill then back-fill
        zscore = pd.Series(zscore).ffill().bfill().values

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }