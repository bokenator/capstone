import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict


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
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This implementation uses a simple target-based approach: for each bar we
    compute the desired target position (in number of shares) for each asset
    based on the z-score and hedge ratio and return the difference between the
    target and the current position as the order size (size_type=0 = shares).

    Notes:
    - No lookahead: decisions use zscore[i], hedge_ratio[i] and historical zscore
      (zscore[i-1]) only.
    - Uses flexible multi-asset mode: called for each asset column separately.
    - Returns (np.nan, 0, 0) to indicate no action.
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # If insufficient data, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base share sizing for Asset A (1 unit equivalent)
    shares_a = notional_per_leg / price_a
    # Asset B scaled by hedge ratio (could be negative)
    shares_b = hr * shares_a

    # Determine target positions (in shares) for each asset
    target_a = 0.0
    target_b = 0.0

    # Stop-loss: if extreme divergence, close positions
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry signals
        if z > entry_threshold:
            # Short A, Long B (B scaled by hedge ratio)
            target_a = -shares_a
            target_b = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +shares_a
            target_b = -shares_b
        else:
            # Possibly exit when z crosses zero
            # Check crossing: only if we have past value
            if i > 0:
                prev_z = zscore[i - 1]
                # If we have an open position and z crossed zero, close
                if not np.isnan(prev_z) and prev_z * z < 0:
                    target_a = 0.0
                    target_b = 0.0
                else:
                    # No change to targets -> keep existing positions
                    return (np.nan, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Select target for this column
    target = target_a if col == 0 else target_b

    # Compute order size as difference between target and current position
    size = target - pos_now

    # If size is effectively zero, do nothing
    if np.isclose(size, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return number of shares to trade (size_type=0 => amount/shares), allow both directions
    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame | np.ndarray | pd.Series,
    asset_b: pd.DataFrame | np.ndarray | pd.Series,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute hedge ratio and z-score for a pairs trading strategy.

    This function is robust to inputs being numpy arrays, pandas Series, or
    DataFrames with a 'close' column. It uses past data only (no lookahead).

    Implementation details:
    - Rolling OLS (slope only) computed on past `min(i+1, hedge_lookback)` bars.
      If there are fewer than 2 valid points, the hedge ratio carries forward the
      last value (or defaults to 1.0 at start).
    - Spread = close_a - hedge_ratio * close_b
    - Z-score uses rolling mean/std with window up to `zscore_lookback`. Std uses
      ddof=0 to avoid NaNs for small windows. If std==0, z-score is set to 0.

    Returns a dict with numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Helper to extract close price array from supported inputs
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            arr = x["close"].values
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            # Try to coerce
            try:
                arr = np.asarray(x)
            except Exception:
                raise ValueError("Unsupported input type for price series")
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if close_a.shape != close_b.shape:
        raise ValueError("Asset price arrays must have the same shape")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio (slope). Use past data only; allow shorter windows
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start : i + 1]
        y = close_a[start : i + 1]

        # Filter out NaNs in the window
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_f = x[mask]
        y_f = y[mask]

        if x_f.size >= 2:
            # Compute OLS slope (y = slope * x + intercept)
            try:
                slope, _, _, _, _ = stats.linregress(x_f, y_f)
                if np.isnan(slope):
                    # fallback to previous ratio or 1.0
                    hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0
                else:
                    hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0
        else:
            # Not enough points: carry forward previous ratio or default to 1.0
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 and not np.isnan(hedge_ratio[i - 1]) else 1.0

    # Compute spread using current hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std (past values only). Use min_periods=1 to avoid long NaN warmups
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().values
    # Use ddof=0 to compute population std (avoids NaN for single observation)
    try:
        spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values
    except TypeError:
        # Older pandas might not support ddof in rolling.std; fall back to default (ddof=1)
        spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std().values

    # Compute z-score, handling zero std
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(spread[i]) or np.isnan(spread_mean[i]) or np.isnan(spread_std[i]):
            zscore[i] = np.nan
        elif spread_std[i] == 0:
            zscore[i] = 0.0
        else:
            zscore[i] = float((spread[i] - spread_mean[i]) / spread_std[i])

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
