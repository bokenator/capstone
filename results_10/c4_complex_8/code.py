import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Any


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
    # Context values
    i = int(c.i)
    col = int(c.col)

    # Position now might be missing or nan
    pos = getattr(c, 'position_now', 0.0)
    try:
        pos = 0.0 if pos is None else float(pos)
    except Exception:
        pos = 0.0

    # Basic bounds checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    # If z-score not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Validate hedge ratio and prices
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    price_a = close_a[i]
    price_b = close_b[i]

    if np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Protect against zero or negative prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Number of shares per leg based on fixed notional
    shares_a = float(notional_per_leg) / float(price_a)
    # Multiply by hedge ratio to get the corresponding units of B
    shares_b = (float(notional_per_leg) / float(price_b)) * float(hr)

    # Priority 1: Stop-loss (close both legs if |z| > stop_threshold)
    if np.abs(z) > stop_threshold:
        if not np.isnan(pos) and pos != 0.0:
            # Close current position for this asset
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Priority 2: Exit when z crosses the exit_threshold (typically 0.0)
    z_prev = zscore[i - 1] if i > 0 else np.nan
    crossed_to_exit = False
    if not np.isnan(z_prev):
        # Detect crossing of exit_threshold (e.g., zero)
        if (z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold):
            crossed_to_exit = True
    if crossed_to_exit:
        if not np.isnan(pos) and pos != 0.0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic
    # Short A, Long B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            # Asset A: short 'shares_a'
            target_pos = -shares_a
        else:
            # Asset B: long 'shares_b' (hedge ratio applied)
            target_pos = shares_b

        # Place order equal to difference between target and current position
        size = target_pos - pos
        # If no change required, do nothing
        if np.isclose(size, 0.0):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)

    # Entry logic symmetrical: z < -entry_threshold => Long A, Short B
    if z < -entry_threshold:
        if col == 0:
            target_pos = shares_a
        else:
            target_pos = -shares_b

        size = target_pos - pos
        if np.isclose(size, 0.0):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)

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

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or a 1D numpy array of closes)
        asset_b: DataFrame with 'close' column for Asset B (or a 1D numpy array of closes)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close arrays from inputs that may be DataFrame/Series or numpy arrays
    def _extract_close(obj: Any, name: str) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError(f"{name} DataFrame must contain 'close' column")
            arr = obj['close'].values
        elif isinstance(obj, pd.Series):
            arr = obj.values
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            # Allow lists as well
            try:
                arr = np.asarray(obj)
            except Exception:
                raise TypeError(f"{name} must be a pd.DataFrame/Series or a numpy array-like")

        arr = np.asarray(arr, dtype=float).squeeze()
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        return arr

    close_a = _extract_close(asset_a, 'asset_a')
    close_b = _extract_close(asset_b, 'asset_b')

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio
    # The first computed hedge_ratio is stored at index = hedge_lookback (uses slices [i-hedge_lookback:i])
    hlb = int(hedge_lookback)
    if hlb < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hlb, n):
        y = close_a[i - hlb:i]
        x = close_b[i - hlb:i]
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            # Not enough data to run regression
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread and z-score
    spread = close_a - hedge_ratio * close_b

    spread_s = pd.Series(spread)
    # Require full lookback for mean/std to avoid unstable early values
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score safely
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        # Where std is zero or NaN, force zscore to NaN
        zscore[np.isnan(spread_std) | (spread_std == 0)] = np.nan

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }