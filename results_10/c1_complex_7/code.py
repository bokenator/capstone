import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    This function is flexible and accepts either:
    - Two pandas DataFrames each containing a 'close' column, or
    - Two numpy arrays / pandas Series representing close prices.

    Returns a dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.

    Notes:
    - hedge_ratio[i] is computed using OLS on the window close[:, i-hedge_lookback:i]
      (i.e. the window ends at index i-1) and stored at index i. This follows the
      example logic in the prompt.
    - Rolling mean/std for z-score use full-window (min_periods = zscore_lookback).
    - Early/warmup periods produce NaNs.
    """
    # Accept numpy arrays / Series or DataFrames with 'close' column
    # Convert inputs to 1D numpy arrays of dtype float
    if isinstance(asset_a, (np.ndarray, pd.Series)) and isinstance(asset_b, (np.ndarray, pd.Series)):
        close_a = np.asarray(asset_a, dtype=float).flatten()
        close_b = np.asarray(asset_b, dtype=float).flatten()
    elif isinstance(asset_a, pd.DataFrame) and isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_a.columns or 'close' not in asset_b.columns:
            raise ValueError("DataFrames must contain 'close' column as per DATA_SCHEMA.")
        close_a = asset_a['close'].astype(float).values
        close_b = asset_b['close'].astype(float).values
    else:
        # Mixed types are not supported
        raise TypeError(
            "asset_a and asset_b must both be numpy arrays / Series or pandas DataFrames with a 'close' column"
        )

    if close_a.shape != close_b.shape:
        raise ValueError("Asset close arrays must have the same shape")

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Prepare outputs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio. We follow the example: for i in range(hedge_lookback, n):
    # use window x = close_b[i-hedge_lookback:i], y = close_a[i-hedge_lookback:i] and store slope at hedge_ratio[i]
    for i in range(hedge_lookback, n):
        x = close_b[i - hedge_lookback: i]
        y = close_a[i - hedge_lookback: i]

        # Require full window and finite values
        if x.size != hedge_lookback or y.size != hedge_lookback:
            continue
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue

        # If x has zero variance, slope is undefined -> skip
        if np.allclose(x, x[0]):
            # If x is constant, the regression is not defined; leave NaN
            continue

        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratio[i] = slope
        except Exception:
            # Any numerical issue -> leave NaN
            continue

    # Compute spread using hedge_ratio. For indices where hedge_ratio is nan, spread will be nan.
    spread = np.full(n, np.nan, dtype=float)
    finite_hr = np.isfinite(hedge_ratio)
    if finite_hr.any():
        spread[finite_hr] = close_a[finite_hr] - hedge_ratio[finite_hr] * close_b[finite_hr]

    # Rolling mean and std for spread using pandas to leverage rolling implementation
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading (flexible multi-asset mode).

    Implements the strategy described in the prompt:
    - Entry when z-score > entry_threshold: short A, long B
    - Entry when z-score < -entry_threshold: long A, short B
    - Exit when z-score crosses exit_threshold (typically 0.0)
    - Stop-loss when |z-score| > stop_threshold
    - Position sizing: fixed notional_per_leg per leg (dollars), no volatility adjustment

    Returns (size, size_type, direction). Uses size_type=0 (amount, shares) and direction=0 (both).

    Notes:
    - This function is called for each asset (col=0 for A, col=1 for B) on each bar.
    - It returns the delta in shares to apply (target_shares - current_position).
    - If no action is required, returns (np.nan, 0, 0).
    """
    # Basic context
    i = int(getattr(c, 'i'))
    col = int(getattr(c, 'col'))
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Bounds check for index
    if i < 0 or i >= len(zscore) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z = zscore[i]

    # If z-score is NaN we cannot make decisions (except we could still close if position exists,
    # but without a valid zscore we don't know exit/entry points). So do nothing.
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if not (np.isfinite(price_a) and price_a > 0 and np.isfinite(price_b) and price_b > 0):
        return (np.nan, 0, 0)

    # Stop-loss has highest priority: close positions if |z| > stop_threshold
    if np.isfinite(z) and abs(z) > float(stop_threshold):
        if pos_now == 0:
            return (np.nan, 0, 0)
        return (-pos_now, 0, 0)

    # Exit if z-score crosses the exit_threshold (e.g., 0.0)
    exit_cross = False
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_diff = zscore[i - 1] - float(exit_threshold)
        curr_diff = z - float(exit_threshold)
        # Crossing: signs differ (allow equality on current)
        if prev_diff * curr_diff <= 0 and prev_diff != 0:
            exit_cross = True

    if exit_cross:
        if pos_now == 0:
            return (np.nan, 0, 0)
        return (-pos_now, 0, 0)

    # Entry logic requires a valid hedge ratio at this bar
    hr = hedge_ratio[i]
    if not np.isfinite(hr) or hr == 0:
        # Can't size the hedge properly -> do not enter new trades
        return (np.nan, 0, 0)

    # Compute target shares based on fixed notional per leg
    shares_a = notional_per_leg / price_a
    # Use absolute hedge ratio to compute magnitude of B shares; sign is determined by trade direction
    shares_b = notional_per_leg / price_b * abs(hr)

    # Determine signals
    enter_short_a_long_b = z > float(entry_threshold)
    enter_long_a_short_b = z < -float(entry_threshold)

    # Compute target for this column
    target = 0.0
    if enter_short_a_long_b:
        # Short A, Long B
        if col == 0:
            target = -shares_a
        else:
            target = shares_b
    elif enter_long_a_short_b:
        # Long A, Short B
        if col == 0:
            target = shares_a
        else:
            target = -shares_b
    else:
        # No entry/exit/stop condition -> hold
        return (np.nan, 0, 0)

    # Compute delta (order size = target - current position)
    delta = float(target - pos_now)

    # If delta is effectively zero, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Return amount (shares)
    return (delta, 0, 0)
