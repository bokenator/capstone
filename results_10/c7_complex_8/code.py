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

    This function is written for flexible=True (multi-asset) mode and is
    NOT numba-compiled. It is called once per asset (col) per bar.

    Args:
        c: Order context with attributes .i (bar index), .col (0=A,1=B),
           .position_now (current position in shares), .cash_now (cash available)
        close_a, close_b: 1D numpy arrays of close prices
        zscore: 1D numpy array of z-score values
        hedge_ratio: 1D numpy array of rolling hedge ratios
        entry_threshold: threshold to enter (e.g., 2.0)
        exit_threshold: threshold to exit (e.g., 0.0)
        stop_threshold: threshold to stop (e.g., 3.0)
        notional_per_leg: fixed dollar notional per leg

    Returns:
        (size, size_type, direction) as described in the prompt.
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))  # current position in shares for this asset

    # Guard indices
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # If zscore or prices are nan, do nothing
    price_a = close_a[i]
    price_b = close_b[i]
    if not np.isfinite(z) or not np.isfinite(price_a) or not np.isfinite(price_b) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    # Determine base share sizing for Asset A (fixed notional per leg)
    # Use absolute number of shares (can be fractional)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    shares_a = notional_per_leg / price_a

    # Apply hedge ratio in units of Asset B per 1 unit of Asset A
    # Use magnitude of hedge ratio for sizing and enforce opposite sign via trade direction
    shares_b = abs(hr) * shares_a

    # Helper to close current position
    if not np.isfinite(pos):
        pos = 0.0

    # STOP-LOSS: if |z| > stop_threshold, close any open position for this asset
    if abs(z) > stop_threshold:
        if pos != 0.0:
            # Close full position
            return (-pos, 0, 0)
        else:
            return (np.nan, 0, 0)

    # EXIT: z-score crossing zero or reaching exit_threshold
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if i > 0 and np.isfinite(prev_z):
        # Detect sign crossing (including touching zero)
        if (prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0):
            crossed_zero = True

    if (crossed_zero or (np.isfinite(z) and abs(z) <= exit_threshold)) and pos != 0.0:
        # Close full position when exit condition met
        return (-pos, 0, 0)

    # ENTRY: only when currently flat
    if pos == 0.0:
        # Long/Short signals based on z-score
        # Avoid entering if z is beyond stop threshold (handled above)
        if z > entry_threshold:
            # Short Asset A, Long Asset B
            if col == 0:
                # Asset A: sell shares_a
                return (-shares_a, 0, 0)
            else:
                # Asset B: buy shares_b
                return (shares_b, 0, 0)
        elif z < -entry_threshold:
            # Long Asset A, Short Asset B
            if col == 0:
                return (shares_a, 0, 0)
            else:
                return (-shares_b, 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a,
    asset_b,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs strategy.

    This function accepts either pandas DataFrame/Series with a 'close' column
    or plain numpy arrays of close prices.

    Args:
        asset_a: DataFrame/Series/ndarray for Asset A (expects 'close' column if DataFrame)
        asset_b: DataFrame/Series/ndarray for Asset B
        hedge_lookback: lookback window for rolling OLS
        zscore_lookback: lookback window for z-score mean/std

    Returns:
        dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Extract close arrays from inputs (support flexible input types)
    def _extract_close(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            if 'close' in x:
                arr = x['close'].values
            else:
                # If DataFrame/Series doesn't have 'close', try to convert directly
                arr = np.asarray(x).squeeze()
        else:
            arr = np.asarray(x).squeeze()
        if arr.ndim != 1:
            raise ValueError('Input close price series must be 1D')
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset price arrays must have the same length')

    n = len(close_a)

    # Initialize hedge_ratio with previous-forward filling strategy to avoid long NaN prefixes
    hedge_ratio = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # dynamic window: use up to hedge_lookback most recent observations including current
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Remove pairs with NaN
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data to estimate; carry forward previous value if available
            if i > 0:
                hedge_ratio[i] = hedge_ratio[i - 1]
            else:
                hedge_ratio[i] = 1.0  # default neutral hedge
            continue

        # Compute OLS slope (regress y on x)
        try:
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            # If slope is finite, assign it; otherwise carry forward
            if np.isfinite(slope):
                hedge_ratio[i] = slope
            else:
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 1.0
        except Exception:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 1.0

    # Compute spread using hedge ratio (use elementwise multiplication)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score. Use min_periods=1 and ddof=0 to avoid NaNs
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Avoid division by zero: where std is zero, set zscore to 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        zscore = np.where(~np.isfinite(zscore), 0.0, zscore)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
