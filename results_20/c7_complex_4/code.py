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
    Generate orders for a pairs trading strategy (flexible multi-asset mode).

    The function computes target sizes for Asset A (col==0) and Asset B (col==1)
    based on the current z-score and hedge ratio. It returns the order required
    to move the current position to the target position (size = target - current).

    Rules implemented:
      - Use prior hedge_ratio[i] and current prices to compute base share sizing
        as: base_shares_a = notional_per_leg / price_a
      - Target positions (in shares):
          * If z > entry_threshold: short A, long B (B scaled by hedge ratio)
          * If z < -entry_threshold: long A, short B
          * If z crosses 0 (sign change) OR |z| > stop_threshold: close both legs
      - Position sizing for B is set to: target_B = -hedge_ratio * target_A
        (this makes B scaled by hedge_ratio relative to A)

    Notes:
      - No action is returned as (np.nan, 0, 0)
      - Orders are returned in shares (size_type=0)
      - This function does not use numba and assumes it will be wrapped by a
        flexible order function wrapper (as provided in the backtest harness).
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    pos = float(getattr(c, "position_now", 0.0))

    # Safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Read z-score and hedge ratio at current bar
    if i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Get current prices
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan

    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0.0 or price_b <= 0.0:
        return (np.nan, 0, 0)

    # Base shares for Asset A (positive number)
    base_shares_a = float(notional_per_leg) / price_a

    # Determine whether we should close due to stop-loss or mean reversion (crossing zero)
    should_close = False
    # Stop-loss
    if abs(z) > float(stop_threshold):
        should_close = True
    # Crossing zero: check previous z-score sign
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        if prev_z * z < 0:
            should_close = True

    # Compute target positions (in shares) for both assets
    target_a = None
    target_b = None

    if should_close:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry logic
        if z > float(entry_threshold):
            # Short Asset A, long Asset B
            target_a = -base_shares_a
            target_b = -hr * target_a  # B scaled by hedge ratio
        elif z < -float(entry_threshold):
            # Long Asset A, short Asset B
            target_a = base_shares_a
            target_b = -hr * target_a
        else:
            # No trading signal
            return (np.nan, 0, 0)

    # Select target based on which column this call is for
    desired = target_a if col == 0 else target_b

    # Calculate required order size to move from current pos to desired pos
    size = float(desired - pos)

    # If difference is negligible, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return order in shares (size_type=0), allow both directions
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a,
    asset_b,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute hedge ratio (rolling OLS) and z-score of the spread for pairs trading.

    This function is robust to inputs being either pandas DataFrames/Series with
    a 'close' column or plain numpy arrays.

    Hedge ratio computation (no lookahead): for each index i, the hedge ratio at
    i is computed by regressing Asset A on Asset B using historical data in
    window [i - hedge_lookback, i) (i excluded). If fewer than 2 valid points
    are available, hedge_ratio[i] stays NaN.

    Z-score computation (no lookahead): spread at i is computed as
        spread[i] = close_a[i] - hedge_ratio[i] * close_b[i]
    The rolling mean and std used to compute z-score at i use only past values
    (i.e., they are computed on spread[:i] using a window of length
    zscore_lookback). We use shift(1) so current spread is excluded from the
    statistics.

    Args:
        asset_a: DataFrame with 'close' or array/Series of closes for Asset A
        asset_b: DataFrame with 'close' or array/Series of closes for Asset B
        hedge_lookback: lookback length for OLS hedge ratio (in bars)
        zscore_lookback: lookback length for rolling mean/std of spread

    Returns:
        Dict containing numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Helper to extract close arrays from different input types
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x["close"].astype(float).to_numpy()
        if isinstance(x, pd.Series):
            return x.astype(float).to_numpy()
        # Assume numpy-like
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Input close price arrays must be 1-dimensional")
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset A and Asset B must have the same length")

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (no lookahead): use data in [i - hedge_lookback, i)
    for i in range(n):
        start = max(0, i - int(hedge_lookback))
        end = i  # exclude current bar
        if end - start < 2:
            # Not enough data to estimate slope reliably
            hedge_ratio[i] = np.nan
            continue
        x = close_b[start:end]
        y = close_a[start:end]
        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread (may contain NaNs where hedge_ratio is NaN)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score: use only past values (exclude current) -> shift(1)
    spread_series = pd.Series(spread)

    # Use full lookback (no partial windows) to avoid noisy early signals
    spread_mean = spread_series.shift(1).rolling(int(zscore_lookback), min_periods=int(zscore_lookback)).mean().to_numpy()
    spread_std = spread_series.shift(1).rolling(int(zscore_lookback), min_periods=int(zscore_lookback)).std(ddof=1).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
