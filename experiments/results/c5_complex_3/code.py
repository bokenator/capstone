import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple, Union


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading (flexible multi-asset mode).

    Args:
        c: OrderContext-like object with attributes i (index), col (0=A,1=B), position_now, cash_now
        close_a: Array of close prices for Asset A
        close_b: Array of close prices for Asset B
        zscore: Array of z-score values for the spread
        hedge_ratio: Array of hedge ratios (beta)
        entry_threshold: Z-score threshold to enter (e.g., 2.0)
        exit_threshold: Z-score threshold to exit (e.g., 0.0)
        stop_threshold: Z-score threshold for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, "position_now", 0.0))

    # Bounds check
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Extract current indicators/prices
    try:
        z = float(zscore[i])
    except Exception:
        return (np.nan, 0, 0)

    try:
        hr = float(hedge_ratio[i])
    except Exception:
        return (np.nan, 0, 0)

    # If indicators are NaN, do nothing
    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    # Determine base shares for 1 unit of Asset A scaled by notional
    # This makes Asset B shares proportional to hedge_ratio * shares_a
    base_shares_a = notional_per_leg / price_a
    shares_a = base_shares_a
    shares_b = hr * base_shares_a

    # Helper to check if we are flat (accounting for tiny numerical noise)
    def is_flat(x: float) -> bool:
        return abs(x) < 1e-8

    # 1) Stop-loss (highest priority): close if |z| > stop_threshold
    if abs(z) > stop_threshold:
        if not is_flat(pos_now):
            # Close this leg
            return (-pos_now, 0, 0)
        else:
            return (np.nan, 0, 0)

    # 2) Exit on mean reversion: z-score crossing the exit_threshold (e.g., 0.0)
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        # Crossing from above to below or below to above
        crossed_down = (prev_z > exit_threshold) and (z <= exit_threshold)
        crossed_up = (prev_z < exit_threshold) and (z >= exit_threshold)
        if (crossed_down or crossed_up) and (not is_flat(pos_now)):
            return (-pos_now, 0, 0)

    # 3) Entry logic: only enter if currently flat for this asset
    if is_flat(pos_now):
        # Short spread: z > entry_threshold -> short A, long B
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B (scaled by hedge ratio)
                return (shares_b, 0, 0)

        # Long spread: z < -entry_threshold -> long A, short B
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B
                return (-shares_b, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio (rolling OLS) and z-score of the spread.

    This function is defensive to inputs of differing lengths: it aligns both
    assets to the shortest available length to avoid lookahead when one asset
    has been truncated.

    Args:
        asset_a: DataFrame with 'close' column or ndarray of close prices for Asset A
        asset_b: DataFrame with 'close' column or ndarray of close prices for Asset B
        hedge_lookback: Lookback period for rolling regression (used as max window)
        zscore_lookback: Lookback period for z-score rolling mean/std

    Returns:
        Dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """

    # Extract close arrays from DataFrame or accept raw arrays
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].to_numpy(dtype=float)
    else:
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].to_numpy(dtype=float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    # Align lengths to the shortest input to avoid lookahead when one series is truncated
    n = min(len(close_a), len(close_b))
    close_a = close_a[:n]
    close_b = close_b[:n]

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression (no lookahead): use data up to and including current index,
    # with a maximum window of `hedge_lookback`. For the earliest bars we use the available
    # history (min 2 points required to compute slope).
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Require at least two finite points to compute slope
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
                hedge_ratio[i] = slope
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread: A - beta * B
    spread = close_a - hedge_ratio * close_b

    # Rolling z-score with min_periods=1 to avoid long NaN warmups
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use ddof=0 (population std) for stability; replace zeros to avoid division by zero
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Safe z-score calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Replace non-finite z-scores (inf/nan) with 0.0 to keep signals clean after warmup
    zscore = np.where(np.isfinite(zscore), zscore, 0.0)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
