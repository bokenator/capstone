import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Any


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

    This implementation is non-numba and uses flexible multi-asset mode.

    Strategy logic summary:
    - When zscore > entry_threshold: short Asset A (1 unit), long Asset B (hedge_ratio * unit)
    - When zscore < -entry_threshold: long Asset A, short Asset B
    - Exit when zscore crosses zero or when |zscore| > stop_threshold (stop-loss)
    - Position sizing: shares_a = notional_per_leg / price_a; position_b = -hedge_ratio * position_a

    Returns a tuple: (size, size_type, direction)
    - size_type: 0 = Amount (shares)
    - direction: 0 = Both
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Guard array lengths
    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    # If required inputs are NaN, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan

    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Size (shares) for asset A based on fixed notional per leg
    shares_a = notional_per_leg / price_a

    # Previous z for zero-cross detection
    prev_z = float(zscore[i - 1]) if i >= 1 and i - 1 < len(zscore) else np.nan

    # Determine desired target positions (in shares) for this bar
    target_a = None  # None means no action (keep existing)

    # Stop-loss: immediate close both legs if breached
    if np.isfinite(z) and abs(z) > stop_threshold:
        target_a = 0.0
    else:
        # Zero-cross exit: if z crosses zero (sign change), request close (only if currently in a position)
        if np.isfinite(prev_z) and (prev_z * z < 0):
            # Only issue close if we currently have a position for this asset
            if pos_now != 0:
                target_a = 0.0
            else:
                # We aren't in a position on this leg; no action
                return (np.nan, 0, 0)
        else:
            # Entry conditions
            if z > entry_threshold:
                # Short A, long B
                target_a = -shares_a
            elif z < -entry_threshold:
                # Long A, short B
                target_a = shares_a
            else:
                # No new signal and no stop/exit: do nothing
                return (np.nan, 0, 0)

    # Compute corresponding target for asset B using hedge ratio
    # Position for B is chosen to hedge A: pos_b = -hr * pos_a
    # This preserves the relationship pos_b ≈ -hedge_ratio * pos_a
    if target_a is None:
        return (np.nan, 0, 0)

    target_b = -hr * target_a

    # Select target for the current column
    target = target_a if col == 0 else target_b

    # Compute order size as delta of shares
    size = float(target - pos_now)

    # If size is effectively zero, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    # Return amount (shares)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for pairs trading.

    Accepts either pandas objects with a 'close' column or raw numpy arrays.

    This implementation uses an expanding/rolling OLS: for each time i we run OLS on the
    most recent up-to-"hedge_lookback" observations available (i.e., if fewer than
    hedge_lookback observations exist yet, we regress on all available past data).
    This avoids long NaN periods at the start while preserving causality (no lookahead).

    Returns a dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    Each value is a numpy array of length N (same as inputs).
    """

    # Helper to extract close numpy array from various input types
    def _to_close_array(x: Any) -> np.ndarray:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if isinstance(x, pd.DataFrame):
                if "close" in x.columns:
                    arr = x["close"].to_numpy(dtype=float)
                else:
                    # fall back to first column
                    arr = x.iloc[:, 0].to_numpy(dtype=float)
            else:
                arr = x.to_numpy(dtype=float)
        else:
            # Assume array-like
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError("Input price arrays must have the same shape")

    n = len(close_a)

    # Allocate hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # For each time index i, regress using available past data up to length 'hedge_lookback'
    L = int(max(1, int(hedge_lookback)))
    for i in range(0, n):
        start = max(0, i - L + 1)
        y = close_a[start : i + 1]
        x = close_b[start : i + 1]

        # Need at least 2 points to run a regression
        if len(x) < 2:
            hedge_ratio[i] = np.nan
            continue

        # Skip windows with NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x is constant, slope is undefined
        if np.allclose(x, x[0]):
            hedge_ratio[i] = np.nan
            continue

        try:
            slope, _, _, _, _ = stats.linregress(x, y)
        except Exception:
            slope = np.nan

        hedge_ratio[i] = float(slope)

    # Compute spread: spread = A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean/std for z-score. Use min_periods=zscore_lookback to ensure meaningful std
    window = int(max(1, int(zscore_lookback)))
    spread_series = pd.Series(spread)

    spread_mean = spread_series.rolling(window=window, min_periods=window).mean().to_numpy(dtype=float)
    spread_std = spread_series.rolling(window=window, min_periods=window).std().to_numpy(dtype=float)

    # Prevent division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = (spread - spread_mean) / spread_std

    # Return arrays of expected length
    return {
        "close_a": close_a.astype(float),
        "close_b": close_b.astype(float),
        "hedge_ratio": hedge_ratio.astype(float),
        "zscore": np.asarray(zscore, dtype=float),
    }