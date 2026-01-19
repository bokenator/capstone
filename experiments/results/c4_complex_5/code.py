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
    notional_per_leg: float = 10000.0,
) -> tuple:
    """
    Order function for a pairs trading strategy (flexible multi-asset).

    Notes:
    - This function is called once per asset per bar by the wrapper in the
      backtest runner. The wrapper provides a simulated context `c` that
      contains `i` (bar index), `col` (asset column: 0=A, 1=B),
      `position_now` (current position in shares) and `cash_now`.
    - It returns a tuple (size, size_type, direction) where size is the
      number of shares (positive=buy, negative=sell), size_type=0 (Amount),
      and direction=0 (Both allowed).

    Strategy summary implemented here:
    - Hedge ratio: provided externally (rolling OLS slope)
    - Spread z-score: provided externally
    - Entry: z > entry_threshold -> short A, long B (hedge units)
             z < -entry_threshold -> long A, short B
    - Exit: z crosses 0.0 -> close both legs
    - Stop-loss: |z| > stop_threshold -> close both legs
    - Position sizing: fixed notional_per_leg per leg

    Args:
        c: OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: numpy array of asset A close prices
        close_b: numpy array of asset B close prices
        zscore: numpy array of spread z-scores
        hedge_ratio: numpy array of rolling hedge ratios
        entry_threshold: threshold to enter (e.g., 2.0)
        exit_threshold: threshold to exit (unused directly; we use zero-crossing)
        stop_threshold: threshold for stop-loss (e.g., 3.0)
        notional_per_leg: fixed dollar exposure per leg

    Returns:
        (size, size_type, direction)
    """
    i = int(getattr(c, 'i', 0))
    col = int(getattr(c, 'col', 0))

    # Current position (in shares). Ensure numeric and finite.
    pos_now = getattr(c, 'position_now', 0.0)
    try:
        pos = float(pos_now) if np.isfinite(pos_now) else 0.0
    except Exception:
        pos = 0.0

    # Basic bounds checking
    if i < 0:
        return (np.nan, 0, 0)

    # Guard array lengths
    n = len(zscore) if zscore is not None else 0
    if i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    # Require valid zscore and hedge ratio for decisions
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    hr = hedge_ratio[i] if (hedge_ratio is not None and i < len(hedge_ratio)) else np.nan
    price_a = close_a[i] if (close_a is not None and i < len(close_a)) else np.nan
    price_b = close_b[i] if (close_b is not None and i < len(close_b)) else np.nan

    if not (np.isfinite(price_a) and np.isfinite(price_b) and np.isfinite(hr)):
        return (np.nan, 0, 0)

    # Avoid division by zero
    if price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    # Compute desired share counts based on fixed notional per leg
    shares_a = float(notional_per_leg / price_a)
    # Follow prompt's example: scale B by hedge ratio
    shares_b = float((notional_per_leg / price_b) * hr)

    # Stop-loss: if |z| > stop_threshold, close any position immediately
    if np.abs(z) > stop_threshold:
        if pos != 0:
            # Close the current asset position
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on zero crossing of z-score: close both legs when sign flips
    if i > 0 and np.isfinite(zscore[i - 1]) and np.isfinite(zscore[i]):
        if (zscore[i - 1] * z) < 0:
            if pos != 0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # Entry logic: only open when we have no current position for this asset
    # Use a small tolerance for zero comparison
    if np.abs(pos) < 1e-8:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B (hedged units)
                return (shares_b, 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                return (shares_a, 0, 0)
            else:
                return (-shares_b, 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> dict:
    """
    Compute hedge ratio (rolling OLS slope) and z-score of the spread.

    This function accepts either pandas DataFrames with a 'close' column or
    numpy arrays / array-like close price series. It returns a dictionary
    with numpy arrays for 'close_a', 'close_b', 'hedge_ratio', and 'zscore'.

    Args:
        asset_a: DataFrame (with 'close') or array-like close prices for Asset A
        asset_b: DataFrame (with 'close') or array-like close prices for Asset B
        hedge_lookback: lookback window for rolling OLS hedge ratio
        zscore_lookback: lookback window for rolling mean/std of spread

    Returns:
        dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Extract close arrays from DataFrame or accept array-like input
    def _extract_close(obj):
        # DataFrame with 'close' column
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                raise ValueError("DataFrame must contain 'close' column")
            return obj['close'].values
        # pandas Series
        if isinstance(obj, pd.Series):
            return obj.values
        # numpy array or array-like
        return np.array(obj, dtype=float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Initialize hedge ratio with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: for each window compute slope of regression price_a ~ slope * price_b
    # The slope is the hedge ratio.
    # Use scipy.stats.linregress (fully qualified) as required.
    for i in range(hedge_lookback, n):
        x = close_b[i - hedge_lookback: i]
        y = close_a[i - hedge_lookback: i]
        # Require finite values in the window
        if (np.sum(np.isfinite(x)) != hedge_lookback) or (np.sum(np.isfinite(y)) != hedge_lookback):
            continue
        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            # In case regression fails for numerical reasons, leave NaN
            hedge_ratio[i] = np.nan

    # Compute spread using the rolling hedge ratio
    # Wherever hedge_ratio is NaN, spread will be NaN
    spread = np.full(n, np.nan, dtype=float)
    # Use vectorized computation but guard NaNs
    valid_hr = np.isfinite(hedge_ratio)
    if np.sum(valid_hr) > 0:
        # For indices where hedge ratio is finite compute spread
        idx = np.where(valid_hr)
        spread[idx] = close_a[idx] - hedge_ratio[idx] * close_b[idx]

    # Rolling mean and std of spread using pandas rolling API
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values

    # z-score with guarding for zero std
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    if np.sum(valid) > 0:
        idx2 = np.where(valid)
        zscore[idx2] = (spread[idx2] - spread_mean[idx2]) / spread_std[idx2]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
