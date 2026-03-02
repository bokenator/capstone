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
) -> Tuple[float, int, int]:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A (np.ndarray)
        close_b: Close prices for Asset B (np.ndarray)
        zscore: Z-score of spread array (np.ndarray)
        hedge_ratio: Rolling hedge ratio array (np.ndarray)
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
    pos = float(getattr(c, 'position_now', 0.0))

    # Protect against out-of-bounds
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # Must have valid z-score and hedge ratio to act
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Protect against zero prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base shares using Asset A notional sizing
    # This sizing will make Asset B shares proportional to hedge_ratio * shares_a
    shares_a_base = notional_per_leg / price_a

    # Desired shares for A and B depend on z-score signal
    # We define desired position in shares (positive = long, negative = short)
    desired_a = 0.0
    desired_b = 0.0

    # Stop-loss: if |z| > stop_threshold --> close both legs
    if abs(z) > stop_threshold:
        # Close this asset's position if any
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Exit rule: z crosses 0 (we check previous bar if available)
    prev_z = np.nan
    if i > 0:
        prev_z = float(zscore[i - 1]) if not np.isnan(zscore[i - 1]) else np.nan

    crossed_to_zero = False
    if not np.isnan(prev_z):
        # Crossing zero means sign change between previous and current
        if prev_z * z < 0:
            crossed_to_zero = True
        # Also treat exact hitting of exit_threshold as exit
        if z == exit_threshold:
            crossed_to_zero = True

    if crossed_to_zero:
        if abs(pos) < 1e-12:
            return (np.nan, 0, 0)
        return (-pos, 0, 0)

    # Entry logic
    if z > entry_threshold:
        # Short Asset A, Long Asset B scaled by hedge ratio
        desired_a = -shares_a_base
        # Desired B in shares should be -hr * desired_a
        desired_b = -hr * desired_a
    elif z < -entry_threshold:
        # Long Asset A, Short Asset B scaled by hedge ratio
        desired_a = shares_a_base
        desired_b = -hr * desired_a
    else:
        # No new entry/exit
        return (np.nan, 0, 0)

    # Map desired shares to the current asset column
    desired = desired_a if col == 0 else desired_b

    # Compute order as difference between desired and current position
    order_size = desired - pos

    # Avoid tiny orders
    if abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Return amount in shares (size_type=0), allow both long and short (direction=0)
    return (float(order_size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrames with 'close' column or raw numpy arrays
    # Convert inputs to numpy arrays of floats
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.Series):
            close_a = asset_a.values
        else:
            if 'close' not in asset_a.columns:
                raise KeyError("asset_a must contain 'close' column")
            close_a = asset_a['close'].values
    elif isinstance(asset_a, np.ndarray):
        close_a = asset_a
    else:
        # Try to coerce
        close_a = np.asarray(asset_a)

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.Series):
            close_b = asset_b.values
        else:
            if 'close' not in asset_b.columns:
                raise KeyError("asset_b must contain 'close' column")
            close_b = asset_b['close'].values
    elif isinstance(asset_b, np.ndarray):
        close_b = asset_b
    else:
        close_b = np.asarray(asset_b)

    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.shape != close_b.shape:
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each time i compute slope using past up to hedge_lookback points (no lookahead)
    # Allow smaller windows at the beginning to avoid long NaN warmup (use past-only data)
    hb = max(2, int(hedge_lookback))
    for i in range(2, n):
        window = min(hb, i)
        y = close_a[i - window:i]
        x = close_b[i - window:i]
        # If any NaNs in window, skip regression
        if np.isnan(y).any() or np.isnan(x).any():
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge ratio (elementwise). If hedge_ratio is nan, spread is nan
    spread = np.full(n, np.nan, dtype=float)
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std for z-score. Use pandas rolling with min_periods to require full window
    spread_series = pd.Series(spread)
    sm = spread_series.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).mean().values
    ss = spread_series.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).std().values

    # Compute z-score with safe division
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(spread[i]) or np.isnan(sm[i]) or np.isnan(ss[i]) or ss[i] == 0:
            zscore[i] = np.nan
        else:
            zscore[i] = float((spread[i] - sm[i]) / ss[i])

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
