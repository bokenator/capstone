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
    # Extract context
    i = int(c.i)
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic guards
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # Require valid indicators and prices
    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Position sizing: base shares for Asset A
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg) / price_a

    # Determine previous z-score for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Stop-loss: |z| > stop_threshold -> close any existing positions
    if abs(z) > float(stop_threshold):
        if pos_now != 0:
            # Close this asset position
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Exit on crossing zero: detect sign change across zero (strict crossing)
    crossed_zero = False
    if not np.isnan(prev_z):
        if (prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0):
            crossed_zero = True

    if crossed_zero:
        if pos_now != 0:
            # Close this asset position
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Entry conditions
    # Only enter when |z| exceeds entry_threshold
    target_pos_a = 0.0
    if z > float(entry_threshold):
        # Short A, Long B (A negative, B positive scaled by hedge ratio)
        target_pos_a = -shares_a
    elif z < -float(entry_threshold):
        # Long A, Short B
        target_pos_a = shares_a
    else:
        # No new entries
        return (np.nan, 0, 0)

    # Compute target for B based on hedge ratio: pos_b = -hedge_ratio * pos_a
    target_pos_b = -hr * target_pos_a

    # Decide which asset we're ordering for
    if col == 0:
        # Asset A
        size = target_pos_a - pos_now
        # If already at target (within tolerance), do nothing
        if np.isclose(size, 0.0, atol=1e-8):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)
    else:
        # Asset B
        size = target_pos_b - pos_now
        if np.isclose(size, 0.0, atol=1e-8):
            return (np.nan, 0, 0)
        return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B OR numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close price array from input
    def _extract_close(obj: Any) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame must contain 'close' column")
            arr = obj['close'].values
        elif isinstance(obj, pd.Series):
            arr = obj.values
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            # Try to coerce
            arr = np.asarray(obj)
        # Ensure 1D float array
        arr = arr.astype(float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no lookahead): for each t compute slope using data up to t (inclusive)
    for i in range(n):
        # Determine window (use available data up to hedge_lookback)
        window_len = int(min(hedge_lookback, i + 1))
        if window_len < 2:
            # Not enough data to compute regression
            hedge_ratio[i] = np.nan
            continue
        start = i - window_len + 1
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]
        # Remove NaNs within window
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score; use min_periods=1 to allow early zscores
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std

    # Where std is zero or NaN, set zscore to 0 (no signal) to avoid infinities
    zero_std_mask = (np.isnan(spread_std) | (np.isclose(spread_std, 0.0)))
    zscore = np.where(zero_std_mask, 0.0, zscore)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
