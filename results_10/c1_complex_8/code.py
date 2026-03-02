import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Tuple, Dict


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

    This implementation uses flexible=True multi-asset mode (no numba).

    Args:
        c: vectorbt OrderContext-like object with attributes:
            - c.i: current bar index (int)
            - c.col: asset column (0=Asset A, 1=Asset B)
            - c.position_now: current position size for this asset (float)
            - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A (1D array)
        close_b: Close prices for Asset B (1D array)
        zscore: Z-score array of spread
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        (size, size_type, direction)
        - size: float (positive=buy, negative=sell)
        - size_type: int (0=Amount/shares, 1=Value($), 2=Percent)
        - direction: int (0=Both, 1=LongOnly, 2=ShortOnly)

    Notes:
        - Orders are expressed in number of shares (size_type=0).
        - Target sizing: we scale Asset A to `notional_per_leg` (shares_a = notional/price_a)
          and set Asset B shares proportional to the hedge ratio: shares_b = hedge_ratio * shares_a.
        - Entry: z > entry_threshold => Short A, Long B
                 z < -entry_threshold => Long A, Short B
        - Exit: z crosses exit_threshold (typically 0.0) OR |z| > stop_threshold => close both legs
    """
    i = int(getattr(c, "i", None))
    col = int(getattr(c, "col", 0))
    pos = float(getattr(c, "position_now", 0.0))

    # Basic bounds check
    if i is None:
        return (np.nan, 0, 0)

    if not (0 <= i < len(zscore)):
        return (np.nan, 0, 0)

    # Read indicators/prices for this bar
    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    # Require valid numbers to act
    if any(np.isnan(x) for x in (z, hr, price_a, price_b)):
        return (np.nan, 0, 0)

    # Avoid division by zero / nonsensical prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine target shares based on notional per leg and hedge ratio
    shares_a = float(notional_per_leg) / float(price_a)
    shares_b = float(hr) * shares_a

    # Determine exit (stop-loss or crossing exit_threshold)
    stop_loss = abs(z) > float(stop_threshold)

    crossed_exit = False
    if i > 0:
        prev = zscore[i - 1]
        # Only consider crossing when previous is finite
        if not np.isnan(prev):
            # Cross above or below the exit threshold
            if (prev > exit_threshold and z <= exit_threshold) or (prev < exit_threshold and z >= exit_threshold):
                crossed_exit = True

    # Determine trade target for this asset (in shares)
    target_shares = None

    if stop_loss or crossed_exit:
        # Close position
        target_shares = 0.0
    else:
        # Entry conditions
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                target_shares = -shares_a
            else:
                target_shares = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                target_shares = +shares_a
            else:
                target_shares = -shares_b
        else:
            # No action (we only enter/exit on thresholds)
            return (np.nan, 0, 0)

    # Compute required order size to move from current pos to target
    size = float(target_shares) - float(pos)

    # If size is effectively zero, do nothing
    if np.isclose(size, 0.0, atol=1e-9):
        return (np.nan, 0, 0)

    # Return number of shares to trade (size_type=0 -> Amount/shares), allow both directions
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame or array-like with 'close' values for Asset A
        asset_b: DataFrame or array-like with 'close' values for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with keys:
            - 'close_a': np.ndarray
            - 'close_b': np.ndarray
            - 'hedge_ratio': np.ndarray
            - 'zscore': np.ndarray

    Notes:
        - The function is robust to inputs being pd.DataFrame with 'close' column,
          pd.Series, list, or numpy arrays.
        - Rolling regression uses scipy.stats.linregress on non-NaN slices.
        - Rolling z-score uses pandas rolling mean/std with min_periods equal to lookback
          to avoid spurious small-sample z-scores.
    """
    # Helper to extract close array from various input types
    def _extract_close(x):
        # If DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            arr = x['close'].astype(float).values
            return arr
        # If Series
        if isinstance(x, pd.Series):
            return x.astype(float).values
        # If numpy array or list-like
        arr = np.asarray(x)
        if arr.ndim == 0:
            return np.array([float(arr)])
        if arr.ndim > 1:
            # If passed a 2D array of shape (n,1) or (n,), try to flatten
            arr = arr.ravel()
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError("Asset A and Asset B must have the same shape/length")

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS to compute hedge ratio (slope of regression of A ~ B)
    # Start filling hedge_ratio at index = hedge_lookback
    for i in range(hedge_lookback, n):
        start = i - hedge_lookback
        end = i
        y = close_a[start:end]
        x = close_b[start:end]

        # Drop NaNs within window
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) < 2:
            # Not enough data to compute regression
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread: A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)

    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use ddof=0 for population std to be stable with small samples
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid_idx = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_idx] = (spread[valid_idx] - spread_mean[valid_idx]) / spread_std[valid_idx]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
