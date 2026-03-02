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
    """
    # Extract context
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B

    # Validate indices
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    h = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    # If z or hedge ratio not available, do nothing
    if np.isnan(z) or np.isnan(h):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Current position for this asset
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Compute basic share sizing (amount of shares for Asset A)
    # Use notional_per_leg to determine base shares for Asset A
    shares_a = float(notional_per_leg) / price_a

    # Determine desired positions (in shares) for each asset based on zscore
    # Default desired is to do nothing (None)
    desired_a = None
    desired_b = None

    # Stop-loss: close if |z| > stop_threshold
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Check for exit: crossing zero or within exit threshold
        z_prev = float(zscore[i - 1]) if i >= 1 and np.isfinite(zscore[i - 1]) else np.nan
        crossed_zero = False
        if not np.isnan(z_prev):
            if z_prev * z < 0:
                crossed_zero = True
        if crossed_zero or abs(z) <= exit_threshold:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short Asset A, Long Asset B
                desired_a = -shares_a
                # Apply hedge ratio so that pos_b ~= -hedge_ratio * pos_a
                desired_b = -h * desired_a
            elif z < -entry_threshold:
                # Long Asset A, Short Asset B
                desired_a = shares_a
                desired_b = -h * desired_a
            else:
                # No trade signal
                desired_a = None
                desired_b = None

    # Decide order for the current column
    if col == 0:
        # Asset A
        if desired_a is None:
            return (np.nan, 0, 0)
        # delta = desired - current
        delta = float(desired_a) - pos_now
        # If no change, do nothing
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (delta, 0, 0)
    else:
        # Asset B
        if desired_b is None:
            return (np.nan, 0, 0)
        delta = float(desired_b) - pos_now
        if abs(delta) < 1e-8:
            return (np.nan, 0, 0)
        return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is written to accept either pandas DataFrames with a 'close'
    column or 1D numpy arrays / pandas Series of close prices. It returns
    numpy arrays for close prices, rolling hedge ratio and z-score of the spread.

    Args:
        asset_a: DataFrame with 'close' column for Asset A (or 1D array/Series)
        asset_b: DataFrame with 'close' column for Asset B (or 1D array/Series)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close arrays from possible input types
    def _extract_close(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            arr = x['close'].to_numpy(dtype=float)
        elif isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Try to convert
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        # If lengths differ, align to the minimum length (no lookahead)
        n = min(len(close_a), len(close_b))
        close_a = close_a[:n]
        close_b = close_b[:n]
    else:
        n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each time i, use up to `hedge_lookback` points ending at i (inclusive)
    # Use min_periods=2 to be able to compute early slopes when data is limited
    for i in range(n):
        win_start = max(0, i - hedge_lookback + 1)
        x_win = close_b[win_start:i + 1]
        y_win = close_a[win_start:i + 1]
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if np.sum(mask) >= 2:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_win[mask], y_win[mask])
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread and rolling z-score using past data only (window ending at i)
    spread = np.full(n, np.nan, dtype=float)
    for i in range(n):
        h = hedge_ratio[i]
        if np.isfinite(h) and np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
            spread[i] = close_a[i] - h * close_b[i]
        else:
            spread[i] = np.nan

    # Use pandas rolling with min_periods=1 to avoid NaNs early; this still ensures no lookahead
    spread_sr = pd.Series(spread)
    spread_mean = spread_sr.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy(dtype=float)
    spread_std = spread_sr.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy(dtype=float)

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        s = spread[i]
        m = spread_mean[i]
        sd = spread_std[i]
        if not np.isfinite(s) or not np.isfinite(m) or not np.isfinite(sd) or sd == 0:
            # If std is zero or any component invalid, set zscore to 0 (no signal)
            if np.isfinite(s) and np.isfinite(m):
                zscore[i] = 0.0
            else:
                zscore[i] = np.nan
        else:
            zscore[i] = (s - m) / sd

    # Final cleanup: ensure no NaN after initial warmup (we avoid heavy NaNs by using expanding windows)
    # Replace any remaining NaNs in hedge_ratio with previous valid value (forward-fill), if any
    if np.all(np.isnan(hedge_ratio)):
        # If entirely NaN (degenerate), set to 0
        hedge_ratio[:] = 0.0
    else:
        # Forward-fill then back-fill remaining
        hr_sr = pd.Series(hedge_ratio)
        hr_sr = hr_sr.fillna(method='ffill').fillna(method='bfill')
        hedge_ratio = hr_sr.to_numpy(dtype=float)

    # For zscore, replace initial NaNs (where spread undefined) with 0 to avoid NaNs after warmup
    zscore = pd.Series(zscore).fillna(0.0).to_numpy(dtype=float)

    return {
        'close_a': close_a.astype(float),
        'close_b': close_b.astype(float),
        'hedge_ratio': hedge_ratio.astype(float),
        'zscore': zscore.astype(float),
    }
