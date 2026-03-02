import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict


def order_func(
    c: Any,
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
    i = int(getattr(c, "i"))  # Current bar index
    col = int(getattr(c, "col"))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))  # Current position for this asset

    # Basic bounds checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If z-score not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices and hedge ratio at current bar
    price_a = float(close_a[i])
    price_b = float(close_b[i])
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # Validate inputs
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)
    if np.isnan(hr):
        # Without hedge ratio we avoid opening new trades
        # Allow closing existing positions based on exits/stop-loss
        pass

    # Shares calculation based on template provided: notional per leg, adjusted by hedge ratio for asset B
    # Protect against division by zero; hr may be nan which will propagate to shares_b
    try:
        shares_a = float(notional_per_leg / price_a)
    except Exception:
        return (np.nan, 0, 0)

    # If hedge ratio is nan or infinite, avoid opening new trades that involve Asset B sizing
    if np.isnan(hr) or not np.isfinite(hr):
        shares_b = np.nan
    else:
        shares_b = float(notional_per_leg / price_b * hr)

    # Previous z-score for crossing detection
    prev_z = float(zscore[i - 1]) if i - 1 >= 0 else np.nan

    # Helper to close current position entirely
    if pos != 0:
        # Stop-loss condition: |z| > stop_threshold -> close
        if abs(z) > stop_threshold:
            return (-pos, 0, 0)

        # Exit on z-score crossing zero (from positive to negative or vice-versa)
        if not np.isnan(prev_z):
            if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
                return (-pos, 0, 0)

    # ENTRY LOGIC
    # Short Asset A, Long Asset B when z > entry_threshold
    if z > entry_threshold:
        if col == 0:
            # Asset A: want to short shares_a
            if pos == 0:
                return (-shares_a, 0, 0)
            # If currently long, reverse (sell enough to close and open short)
            if pos > 0:
                # desired final position = -shares_a; order size = desired - current
                size = -shares_a - pos
                return (float(size), 0, 0)
            # Already short: no action
            return (np.nan, 0, 0)

        elif col == 1:
            # Asset B: want to long shares_b
            # If shares_b not computable, do not open
            if np.isnan(shares_b):
                return (np.nan, 0, 0)
            if pos == 0:
                return (shares_b, 0, 0)
            if pos < 0:
                # currently short, buy enough to close and go long
                size = -pos + shares_b
                return (float(size), 0, 0)
            return (np.nan, 0, 0)

    # z < -entry_threshold: Long Asset A, Short Asset B
    if z < -entry_threshold:
        if col == 0:
            # Asset A: want to long shares_a
            if pos == 0:
                return (shares_a, 0, 0)
            if pos < 0:
                # currently short, buy to close and open long
                size = -pos + shares_a
                return (float(size), 0, 0)
            return (np.nan, 0, 0)

        elif col == 1:
            # Asset B: want to short shares_b
            if np.isnan(shares_b):
                return (np.nan, 0, 0)
            if pos == 0:
                return (-shares_b, 0, 0)
            if pos > 0:
                # currently long, sell to close and open short
                size = - (pos + shares_b)
                return (float(size), 0, 0)
            return (np.nan, 0, 0)

    # No action by default
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
        asset_a: DataFrame with 'close' column for Asset A (or numpy array of closes)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array of closes)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame (with 'close') or raw numpy arrays
    if isinstance(asset_a, np.ndarray):
        close_a = np.asarray(asset_a, dtype=float)
    elif isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = np.asarray(asset_a['close'], dtype=float)
    else:
        # Try to coerce
        close_a = np.asarray(asset_a, dtype=float)

    if isinstance(asset_b, np.ndarray):
        close_b = np.asarray(asset_b, dtype=float)
    elif isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = np.asarray(asset_b['close'], dtype=float)
    else:
        close_b = np.asarray(asset_b, dtype=float)

    if len(close_a) != len(close_b):
        raise ValueError('Input series must have the same length')

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio (slope of regression of A ~ B)
    # Start filling hedge_ratio from index = hedge_lookback (inclusive)
    for end in range(hedge_lookback, n + 1):
        i = end - 1  # current index where we store slope
        start = end - hedge_lookback
        y = close_a[start:end]
        x = close_b[start:end]
        # Filter finite values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        x_f = x[mask]
        y_f = y[mask]
        # Require variance in x
        if np.nanstd(x_f) == 0:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_f, y_f)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio per-bar
    spread = np.full(n, np.nan, dtype=float)
    for i in range(n):
        hr = hedge_ratio[i]
        if np.isnan(hr) or not np.isfinite(hr):
            spread[i] = np.nan
        else:
            a = close_a[i]
            b = close_b[i]
            if not np.isfinite(a) or not np.isfinite(b):
                spread[i] = np.nan
            else:
                spread[i] = a - hr * b

    # Rolling mean and std for z-score
    spread_ser = pd.Series(spread)
    spread_mean = spread_ser.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_ser.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        m = spread_mean[i]
        s = spread_std[i]
        sp = spread[i]
        if np.isfinite(sp) and np.isfinite(m) and np.isfinite(s) and s > 0:
            zscore[i] = (sp - m) / s
        else:
            zscore[i] = np.nan

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
