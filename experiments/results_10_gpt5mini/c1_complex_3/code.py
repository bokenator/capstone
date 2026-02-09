import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This implementation follows the specification:
      - Entry when zscore > entry_threshold or zscore < -entry_threshold
      - Exit when zscore crosses zero or when |zscore| > stop_threshold
      - Position sizing: fixed notional per leg (notional_per_leg)

    Note: This function is written for flexible multi-asset mode; it is called
    once per asset per bar. It returns an order tuple (size, size_type, direction).
    """
    i = int(c.i)
    col = int(c.col)  # 0 = Asset A, 1 = Asset B

    # Defensive checks
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are numpy arrays
    close_a = np.asarray(close_a)
    close_b = np.asarray(close_b)
    zscore = np.asarray(zscore)
    hedge_ratio = np.asarray(hedge_ratio)

    # Bounds check
    if i >= len(zscore) or i >= len(close_a) or i >= len(close_b) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    h = float(hedge_ratio[i])

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(h):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Prices must be positive
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Current position for this asset (number of shares, can be negative for short)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Determine share sizes based on fixed notional per leg.
    # Following the example in the specification, compute:
    #   shares_a = notional_per_leg / price_a
    #   shares_b = notional_per_leg / price_b * hedge_ratio
    # This keeps the notional per leg fixed while scaling the units of Asset B by hedge ratio.
    shares_a = notional_per_leg / price_a
    shares_b = (notional_per_leg / price_b) * h

    # Stop-loss: close positions if |z| exceeds stop_threshold
    if np.isfinite(stop_threshold) and abs(z) > stop_threshold:
        # Close position for this asset
        return (-pos_now, 0, 0)

    # Exit condition: z-score crosses zero => close positions
    # Detect crossing using previous bar sign change
    crossed_zero = False
    if i > 0 and np.isfinite(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        if prev_z != 0 and z != 0 and (prev_z * z) < 0:
            crossed_zero = True
        # Also handle the case where z moved inside the small exit_threshold band
        if abs(z) <= abs(exit_threshold):
            # If previously we were outside the band and now inside, consider it an exit
            if abs(prev_z) > abs(exit_threshold):
                crossed_zero = True

    if crossed_zero:
        return (-pos_now, 0, 0)

    # ENTRY LOGIC
    # If z > entry_threshold: SHORT A, LONG B
    if z > entry_threshold:
        if col == 0:
            # Asset A: go short
            desired = -shares_a
            trade = desired - pos_now
            # If trade is effectively zero, skip
            if abs(trade) < 1e-8:
                return (np.nan, 0, 0)
            return (float(trade), 0, 0)
        else:
            # Asset B: go long (scaled by hedge ratio)
            desired = shares_b
            trade = desired - pos_now
            if abs(trade) < 1e-8:
                return (np.nan, 0, 0)
            return (float(trade), 0, 0)

    # If z < -entry_threshold: LONG A, SHORT B
    if z < -entry_threshold:
        if col == 0:
            # Asset A: go long
            desired = shares_a
            trade = desired - pos_now
            if abs(trade) < 1e-8:
                return (np.nan, 0, 0)
            return (float(trade), 0, 0)
        else:
            # Asset B: go short
            desired = -shares_b
            trade = desired - pos_now
            if abs(trade) < 1e-8:
                return (np.nan, 0, 0)
            return (float(trade), 0, 0)

    # No action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs trading strategy.

    Accepts either DataFrames with a 'close' column or 1D numpy arrays of close prices.

    Returns a dict with keys:
      - 'close_a': np.ndarray
      - 'close_b': np.ndarray
      - 'hedge_ratio': np.ndarray
      - 'zscore': np.ndarray

    Hedge ratio is computed as the rolling OLS slope (y = A, x = B) over hedge_lookback periods.
    Z-score is calculated from spread = A - hedge_ratio * B using rolling mean/std over zscore_lookback.
    """
    # Accept both pd.DataFrame (with 'close') and raw numpy arrays
    if isinstance(asset_a, np.ndarray):
        close_a = np.asarray(asset_a, dtype=float)
    elif isinstance(asset_a, pd.DataFrame):
        if 'close' not in asset_a.columns:
            raise ValueError("asset_a DataFrame must contain 'close' column")
        close_a = asset_a['close'].to_numpy(dtype=float)
    else:
        # Try to coerce
        try:
            close_a = np.asarray(asset_a, dtype=float)
        except Exception:
            raise TypeError("asset_a must be a pandas DataFrame or a numpy array-like")

    if isinstance(asset_b, np.ndarray):
        close_b = np.asarray(asset_b, dtype=float)
    elif isinstance(asset_b, pd.DataFrame):
        if 'close' not in asset_b.columns:
            raise ValueError("asset_b DataFrame must contain 'close' column")
        close_b = asset_b['close'].to_numpy(dtype=float)
    else:
        try:
            close_b = np.asarray(asset_b, dtype=float)
        except Exception:
            raise TypeError("asset_b must be a pandas DataFrame or a numpy array-like")

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("asset_a and asset_b must have the same length")

    n = len(close_a)

    # Initialize hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS to compute hedge ratio (slope). We handle NaNs by dropping them within the window.
    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Mask invalid values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data to compute regression
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    # Where hedge_ratio is NaN, spread should be NaN
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for spread
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    # Compute z-score safely (avoid division by zero)
    zscore = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
