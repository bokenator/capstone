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

    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    # If we don't have a valid z-score, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Prices for sizing
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Hedge ratio must be available for sizing B leg
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    if np.isnan(hr):
        # Can't size B leg properly without hedge ratio; do not act
        return (np.nan, 0, 0)

    # Determine number of shares for Asset A (fixed notional per leg)
    shares_a = float(notional_per_leg) / price_a
    # Size asset B relative to A using hedge ratio (shares scaling)
    shares_b = shares_a * hr

    # Detect previous z for zero-cross detection
    zprev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine desired positions for this bar
    desired_a = None  # in shares (positive = long A, negative = short A)
    desired_b = None  # in shares (positive = long B, negative = short B)

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > float(stop_threshold):
        desired_a = 0.0
        desired_b = 0.0

    # Exit on zero cross: if previous z exists and signs crossed relative to exit_threshold
    elif not np.isnan(zprev):
        crossed_up = (zprev > exit_threshold) and (z <= exit_threshold)
        crossed_down = (zprev < exit_threshold) and (z >= exit_threshold)
        if crossed_up or crossed_down:
            desired_a = 0.0
            desired_b = 0.0

    # Entry conditions
    if desired_a is None:
        if z > float(entry_threshold):
            # Short A, Long B
            desired_a = -shares_a
            desired_b = +shares_b
        elif z < -float(entry_threshold):
            # Long A, Short B
            desired_a = +shares_a
            desired_b = -shares_b

    # If still None, no trading action this bar
    if desired_a is None:
        return (np.nan, 0, 0)

    # Map to current column
    if col == 0:
        order_size = desired_a - pos
    else:
        order_size = desired_b - pos

    # If order is negligibly small, do nothing
    if abs(order_size) < 1e-8:
        return (np.nan, 0, 0)

    # Size type = 0 (Amount = shares), direction = 0 (Both allowed)
    return (float(order_size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A OR a numpy array of close prices
        asset_b: DataFrame with 'close' column for Asset B OR a numpy array of close prices
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """

    # Accept either DataFrame/Series or numpy arrays for inputs
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            arr = x['close'].values.astype(float)
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        else:
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("Input series must have the same length")

    n = len(close_a)

    # Hedge ratio via rolling OLS. For each index i we regress using past data up to i (exclusive)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # We'll allow smaller windows at the beginning (min_periods=2) to avoid long NaN warmup
    for i in range(n):
        start = max(0, i - int(hedge_lookback))
        end = i  # exclusive so we only use past data
        window_x = close_b[start:end]
        window_y = close_a[start:end]
        if len(window_x) >= 2 and not np.all(np.isnan(window_x)) and not np.all(np.isnan(window_y)):
            # Use linregress on past window (no lookahead)
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(window_x, window_y)
                hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # Compute spread using hedge_ratio estimated from past windows
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score. Use min_periods = zscore_lookback to ensure stability.
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).mean().values
    spread_std = spread_series.rolling(window=int(zscore_lookback), min_periods=int(zscore_lookback)).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid_mask] = (spread[valid_mask] - spread_mean[valid_mask]) / spread_std[valid_mask]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
