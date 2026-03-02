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
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic checks
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # Do not act if indicators are not available
    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Sanity price checks
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute base share sizing for Asset A (fixed notional per leg)
    shares_a = notional_per_leg / price_a
    # Apply hedge ratio to determine B shares (so that shares_b ~= hedge_ratio * shares_a)
    shares_b = hr * shares_a

    # Determine previous z for exit crossing detection
    prev_z = np.nan
    if i > 0:
        prev_z = float(zscore[i - 1])

    # Exit conditions
    exit_signal = False
    # Stop-loss: absolute z exceeds stop threshold
    if np.isfinite(z) and abs(z) > stop_threshold:
        exit_signal = True

    # Mean-reversion exit: z-score crosses the exit_threshold (commonly 0.0)
    if not exit_signal and np.isfinite(prev_z):
        try:
            # Crossing detection relative to exit_threshold
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                exit_signal = True
        except Exception:
            pass

    # Determine desired target positions (in shares) for each asset
    target_a = 0.0
    target_b = 0.0

    if exit_signal:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry signals
        if z > entry_threshold:
            # Short A, Long B
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +shares_a
            target_b = -shares_b
        else:
            # No new signal
            return (np.nan, 0, 0)

    # Choose target for this column
    target = target_a if col == 0 else target_b

    # Compute order as delta between target and current position
    delta = float(target - pos)

    # If nothing to do (within a tiny tolerance), return no action
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return order: size is amount (shares), allow both long and short
    return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or numpy array/Series)
        asset_b: DataFrame with 'close' column for Asset B (or numpy array/Series)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Accept either DataFrame/Series or numpy arrays for flexibility
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return x['close'].to_numpy(dtype=float)
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float)
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling OLS hedge ratio using only past data (no lookahead)
    for i in range(n):
        # Use previous data up to index i-1. For i==0, there is no past data.
        start = max(0, i - hedge_lookback)
        end = i  # exclusive: only past values
        if end - start >= 2:
            x = close_b[start:end]
            y = close_a[start:end]
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan
            else:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread using current prices and hedge ratio estimated from past
    spread = np.full(n, np.nan, dtype=float)
    for i in range(n):
        hr = hedge_ratio[i]
        if np.isfinite(hr) and np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
            spread[i] = close_a[i] - hr * close_b[i]
        else:
            spread[i] = np.nan

    # Rolling mean and std for z-score (uses current and past spreads only)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to avoid small-sample NaNs when min_periods met
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    # Where std == 0 but mean is finite, set zscore to 0 (no deviation)
    zero_std_mask = np.isfinite(spread) & np.isfinite(spread_mean) & (spread_std == 0)
    zscore[zero_std_mask] = 0.0

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }