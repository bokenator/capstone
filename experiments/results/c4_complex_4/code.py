import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy.stats


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0
) -> tuple:
    """
    Generate orders for a pairs trading strategy (flexible multi-asset mode).

    This function is designed to be called by a wrapper that invokes it for
    each asset (col) per bar. It returns a tuple (size, size_type, direction)
    where size is in number of shares (size_type=0).

    Args:
        c: Order context with attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current available cash (float)
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array of z-score values of the spread
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: z-score level to enter (e.g., 2.0)
        exit_threshold: z-score level to exit (e.g., 0.0)
        stop_threshold: z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        A tuple (size, size_type, direction) as required by the runner wrapper.
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B

    # Defensive: ensure arrays are long enough
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Get current z-score and hedge ratio
    z = zscore[i] if (0 <= i < len(zscore)) else np.nan
    hr = hedge_ratio[i] if (0 <= i < len(hedge_ratio)) else np.nan

    # If insufficient info, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]

    # Validate prices
    if not (np.isfinite(price_a) and np.isfinite(price_b)):
        return (np.nan, 0, 0)
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute target share sizes based on fixed notional per leg
    shares_a = float(notional_per_leg) / float(price_a)
    # Hedge: number of B shares = hedge_ratio * number of A shares
    shares_b = float(hr) * shares_a

    # Determine previous z to detect crossing of exit threshold (usually 0)
    prev_z = zscore[i - 1] if i > 0 and (i - 1) < len(zscore) else np.nan
    crossed_zero = False
    if np.isfinite(prev_z) and np.isfinite(z):
        if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
            crossed_zero = True

    # Determine targets based on z-score
    target_a = None
    target_b = None

    # Stop-loss: if absolute zscore exceeds stop threshold -> close both
    if np.abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    # Exit: z-score crossed exit_threshold (usually 0)
    elif crossed_zero:
        target_a = 0.0
        target_b = 0.0
    # Entry: short A, long B when z > entry_threshold
    elif z > entry_threshold:
        target_a = -shares_a
        target_b = shares_b
    # Entry: long A, short B when z < -entry_threshold
    elif z < -entry_threshold:
        target_a = shares_a
        target_b = -shares_b
    else:
        # No trading signal
        return (np.nan, 0, 0)

    # Current position for this asset
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now) if np.isfinite(pos_now) else 0.0
    except Exception:
        pos_now = 0.0

    target = target_a if col == 0 else target_b

    # If target is None then no action
    if target is None or not np.isfinite(target):
        return (np.nan, 0, 0)

    # Compute order size (number of shares to trade)
    size = float(target - pos_now)

    # Small tolerance to avoid noise orders
    if np.abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return number of shares (amount), allow both long and short
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> dict:
    """
    Compute rolling hedge ratio (OLS), spread, and z-score for a pairs strategy.

    Accepts either pd.DataFrame with a 'close' column or 1D numpy arrays for
    asset_a and asset_b. Returns a dict with numpy arrays:
      - 'close_a'
      - 'close_b'
      - 'hedge_ratio'
      - 'zscore'

    Args:
        asset_a: DataFrame with 'close' column or 1D numpy array
        asset_b: DataFrame with 'close' column or 1D numpy array
        hedge_lookback: lookback window for rolling OLS (in bars)
        zscore_lookback: window for rolling mean/std when computing z-score

    Returns:
        Dictionary of indicators as numpy arrays.
    """
    # Extract close price arrays from inputs (support DataFrame or ndarray)
    if isinstance(asset_a, pd.DataFrame):
        if "close" not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a["close"].values
    else:
        close_a = np.array(asset_a, dtype=np.float64)

    if isinstance(asset_b, pd.DataFrame):
        if "close" not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b["close"].values
    else:
        close_b = np.array(asset_b, dtype=np.float64)

    if len(close_a) != len(close_b):
        raise ValueError("Input price arrays must have the same length")

    n = len(close_a)

    # Validate lookbacks
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS to estimate hedge ratio (slope of regression y ~ x where y=Price_A, x=Price_B)
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback : i]
        x = close_b[i - hedge_lookback : i]

        # Drop pairs with NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread using the rolling hedge ratio
    spread = np.full(n, np.nan)
    valid_hr = np.isfinite(hedge_ratio)
    valid_prices = np.isfinite(close_a) & np.isfinite(close_b)

    mask_spread = valid_hr & valid_prices
    spread[mask_spread] = close_a[mask_spread] - hedge_ratio[mask_spread] * close_b[mask_spread]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values

    zscore = np.full(n, np.nan)
    valid_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
