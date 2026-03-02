import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple


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
) -> Tuple[float, int, int]:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset order function).

    This function computes target position sizes (in number of shares) for each asset
    based on the z-score of the spread and the rolling hedge ratio. It returns the
    delta (target - current) in shares as the order size (size_type=0 = Amount).

    Args:
        c: Order context object with attributes:
           - c.i: current bar index (int)
           - c.col: current column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float)
        close_a: 1D numpy array of Asset A close prices
        close_b: 1D numpy array of Asset B close prices
        zscore: 1D numpy array of spread z-score
        hedge_ratio: 1D numpy array of rolling hedge ratio
        entry_threshold: z-score level to enter (positive float)
        exit_threshold: z-score level for exit (typically 0.0)
        stop_threshold: absolute z-score for stop-loss
        notional_per_leg: fixed notional in $ per leg

    Returns:
        (size, size_type, direction)
        - size: float number of shares (positive=buy, negative=sell)
        - size_type: int (0=Amount, 1=Value, 2=Percent)
        - direction: int (0=Both)

    Notes:
        - Uses size_type=0 (Amount) and returns the difference between target and
          current position as the order size.
        - If the function decides to take no action, it returns (np.nan, 0, 0).
    """
    # Extract context
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely get current position (number of shares). If not provided, assume 0.
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = 0.0 if pos_now is None else float(pos_now)
    except Exception:
        pos_now = 0.0

    # Basic bounds check
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Read current indicators/prices
    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    # Validate inputs
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target shares (magnitude may be fractional)
    shares_a = float(notional_per_leg / price_a)
    # Preserve sign of hedge_ratio as given (following problem statement)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Determine conditions: stop-loss highest priority, then exit (z crosses exit_threshold), then entry
    # Stop-loss: absolute z-score exceeds stop_threshold
    if np.abs(z) > stop_threshold:
        target = 0.0
    else:
        # Detect z-score crossing the exit_threshold (typically zero)
        z_prev = zscore[i - 1] if i > 0 else np.nan
        cross_zero = False
        if not np.isnan(z_prev):
            # Consider crossing when previous and current are on opposite sides or have reached the exit threshold
            if (z_prev > exit_threshold and z <= exit_threshold) or (z_prev < exit_threshold and z >= exit_threshold):
                cross_zero = True
        if cross_zero:
            target = 0.0
        else:
            # Entry signals
            if z > entry_threshold:
                # Short A, Long B (A -> negative, B -> positive)
                target = -shares_a if col == 0 else shares_b
            elif z < -entry_threshold:
                # Long A, Short B (A -> positive, B -> negative)
                target = shares_a if col == 0 else -shares_b
            else:
                # No signal
                return (np.nan, 0, 0)

    # Compute order as delta in shares
    size = float(target - pos_now)

    # If the delta is effectively zero, do nothing
    if np.isclose(size, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # size_type=0 => Amount (shares); direction=0 => allow both long and short
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio and spread z-score for a pairs trading strategy.

    Args:
        asset_a: DataFrame or array-like containing 'close' column/values for Asset A
        asset_b: DataFrame or array-like containing 'close' column/values for Asset B
        hedge_lookback: lookback window (in bars) for rolling OLS regression to estimate hedge ratio
        zscore_lookback: lookback window for spread mean/std used to compute z-score

    Returns:
        Dictionary with keys:
          - 'close_a': numpy array of Asset A close prices
          - 'close_b': numpy array of Asset B close prices
          - 'hedge_ratio': numpy array of rolling hedge ratios (NaN where not available)
          - 'zscore': numpy array of spread z-scores (NaN where not available)

    Notes:
        - When asset_a/asset_b are DataFrames, only the 'close' column is accessed (per DATA_SCHEMA).
        - Rolling OLS uses scipy.stats.linregress on non-NaN pairs within each window; if fewer than 2
          valid points exist, hedge ratio is left as NaN for that index.
    """
    # Extract close arrays from inputs (allow passing raw numpy arrays as well)
    if isinstance(asset_a, (pd.DataFrame, pd.Series)):
        if isinstance(asset_a, pd.Series):
            close_a = asset_a.values
        else:
            if 'close' not in asset_a.columns:
                raise KeyError("asset_a DataFrame must contain 'close' column")
            close_a = asset_a['close'].values
    elif isinstance(asset_a, np.ndarray):
        close_a = asset_a
    else:
        # Attempt to coerce
        close_a = np.asarray(asset_a)

    if isinstance(asset_b, (pd.DataFrame, pd.Series)):
        if isinstance(asset_b, pd.Series):
            close_b = asset_b.values
        else:
            if 'close' not in asset_b.columns:
                raise KeyError("asset_b DataFrame must contain 'close' column")
            close_b = asset_b['close'].values
    elif isinstance(asset_b, np.ndarray):
        close_b = asset_b
    else:
        close_b = np.asarray(asset_b)

    # Ensure 1D numpy arrays
    close_a = np.asarray(close_a, dtype=float).flatten()
    close_b = np.asarray(close_b, dtype=float).flatten()

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling hedge ratio (slope from OLS of y = close_a ~ x = close_b)
    hedge_ratio = np.full(n, np.nan)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        # Filter out NaNs
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            # Not enough data to estimate
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread using the rolling hedge ratio. Where hedge_ratio is NaN, spread will be NaN.
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use ddof=0 (population std) for stability; pandas default ddof=1, but either is acceptable.
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
