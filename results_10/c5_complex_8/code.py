import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Union, Dict


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
    col = int(getattr(c, 'col', 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, 'position_now', 0.0))  # current position (in shares)

    # Safety checks
    if i < 0:
        return (np.nan, 0, 0)

    # Extract current indicators
    z = float(zscore[i]) if i < len(zscore) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # If zscore or hedge ratio is NaN, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Protect against zero or negative prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute shares for Asset A based on fixed notional
    shares_a = float(notional_per_leg) / price_a
    # Apply hedge ratio to shares of asset B so that shares_b = hedge_ratio * shares_a
    shares_b = shares_a * hr

    # Determine target positions (in shares)
    target_a = None
    target_b = None

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry logic
        if z > entry_threshold:
            # Short A, Long B
            target_a = -shares_a
            target_b = shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = shares_a
            target_b = -shares_b
        else:
            # Check exit on crossing of exit_threshold (usually 0.0)
            if i > 0 and not np.isnan(zscore[i - 1]):
                zprev = float(zscore[i - 1])
                crossed = False
                # Crossing from above to at/below exit_threshold or vice versa
                if (zprev > exit_threshold and z <= exit_threshold) or (zprev < exit_threshold and z >= exit_threshold):
                    crossed = True
                if crossed:
                    target_a = 0.0
                    target_b = 0.0
                else:
                    # No change to targets; do nothing
                    return (np.nan, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Return orders as delta to reach target for the given column
    if col == 0:
        delta = float(target_a) - pos
        if abs(delta) < 1e-9:
            return (np.nan, 0, 0)
        return (delta, 0, 0)
    else:
        delta = float(target_b) - pos
        if abs(delta) < 1e-9:
            return (np.nan, 0, 0)
        return (delta, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A or numpy array of closes
        asset_b: DataFrame with 'close' column for Asset B or numpy array of closes
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close arrays from inputs (support both DataFrame and ndarray)
    if isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise KeyError("asset_a DataFrame must contain 'close' column")
        close_a = np.array(asset_a['close'], dtype=float)
    elif isinstance(asset_a, pd.Series):
        close_a = np.array(asset_a, dtype=float)
    elif isinstance(asset_a, np.ndarray):
        close_a = np.array(asset_a, dtype=float)
    else:
        raise TypeError('asset_a must be a DataFrame, Series, or numpy array')

    if isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise KeyError("asset_b DataFrame must contain 'close' column")
        close_b = np.array(asset_b['close'], dtype=float)
    elif isinstance(asset_b, pd.Series):
        close_b = np.array(asset_b, dtype=float)
    elif isinstance(asset_b, np.ndarray):
        close_b = np.array(asset_b, dtype=float)
    else:
        raise TypeError('asset_b must be a DataFrame, Series, or numpy array')

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each time i, use the trailing window ending at i (inclusive)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Need at least 2 points to run regression
        if x.size < 2:
            # Use previous hedge ratio if available, otherwise default to 1.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0
            continue

        # Handle constant x (zero variance) by falling back to previous ratio
        if np.allclose(x, x[0]):
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0
            continue

        # Perform OLS regression (y = slope * x + intercept)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            if np.isnan(slope):
                raise ValueError('slope is NaN')
            hedge_ratio[i] = float(slope)
        except Exception:
            # Fallback to previous value or 1.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0

    # Compute spread and z-score using trailing windows (inclusive)
    spread = close_a - hedge_ratio * close_b

    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use population std (ddof=0) to be consistent
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Avoid division by zero
    zscore = np.full(n, np.nan, dtype=float)
    nonzero = spread_std > 0
    zscore[nonzero] = (spread[nonzero] - spread_mean[nonzero]) / spread_std[nonzero]
    # For zero std, set zscore to 0 (no dispersion)
    zscore[~nonzero] = 0.0

    return {
        'close_a': np.array(close_a, dtype=float),
        'close_b': np.array(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
