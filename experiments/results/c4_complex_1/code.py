import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy


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
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    Notes:
    - This function is written for flexible (multi-asset) mode and is NOT numba-jitted.
    - The wrapper used by the backtester passes arguments in the order
      (close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold).
      Therefore zscore comes before hedge_ratio in this signature.

    Args:
        c: OrderContext-like object with attributes i (index), col (asset column),
           position_now (current position in shares), cash_now (current cash, optional)
        close_a: numpy array of Asset A close prices
        close_b: numpy array of Asset B close prices
        zscore: numpy array of spread z-scores
        hedge_ratio: numpy array of rolling hedge ratios
        entry_threshold: z-score entry threshold (e.g., 2.0)
        exit_threshold: z-score exit threshold (e.g., 0.0)
        stop_threshold: z-score stop-loss threshold (e.g., 3.0)
        notional_per_leg: fixed dollar notional per leg (default 10000.0)

    Returns:
        (size, size_type, direction) tuple as described in the prompt.
    """
    # Current bar and asset
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B

    # Current position for this asset (in shares)
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Validate arrays lengths
    n = len(zscore) if hasattr(zscore, "__len__") else 0
    if i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    if np.isnan(z) or not np.isfinite(z):
        return (np.nan, 0, 0)

    # Prices and hedge ratio at current bar
    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan
    beta = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    # Validate prices and hedge ratio
    if (not np.isfinite(price_a)) or (not np.isfinite(price_b)) or (not np.isfinite(beta)):
        return (np.nan, 0, 0)

    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine share sizes based on fixed notional per leg
    # shares_a: number of Asset A shares to trade
    # shares_b: number of Asset B shares scaled by hedge ratio (use absolute hedge ratio for sizing)
    try:
        shares_a = float(notional_per_leg) / float(price_a)
    except Exception:
        return (np.nan, 0, 0)

    # Use absolute hedge ratio for sizing so that number of B shares is positive quantity
    hr_abs = float(np.abs(beta)) if np.isfinite(beta) else np.nan
    if not np.isfinite(hr_abs):
        return (np.nan, 0, 0)

    try:
        shares_b = float(notional_per_leg) / float(price_b) * hr_abs
    except Exception:
        return (np.nan, 0, 0)

    # Helper: close this asset's position entirely
    def _close_position() -> tuple:
        if pos_now == 0:
            return (np.nan, 0, 0)
        # Sell (negative) if currently long (positive pos), buy (positive) if currently short (negative pos)
        return (-pos_now, 0, 0)

    # 1) Stop-loss: if we have a position and |z| > stop_threshold -> close
    if np.abs(z) > stop_threshold:
        return _close_position()

    # 2) Exit: z-score crosses exit_threshold (typically 0.0) -> close
    z_prev = zscore[i - 1] if i > 0 else np.nan
    if np.isfinite(z_prev):
        crossed_down = (z_prev > exit_threshold) and (z <= exit_threshold)
        crossed_up = (z_prev < exit_threshold) and (z >= exit_threshold)
        if (crossed_down or crossed_up) and (pos_now != 0):
            return _close_position()

    # 3) Entry: only enter if no existing position for this asset
    if pos_now == 0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B
                return (shares_b, 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B
                return (-shares_b, 0, 0)

    # Otherwise: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with a 'close' column or raw numpy arrays / iterables.

    Returns a dictionary with keys:
      - 'close_a': np.ndarray
      - 'close_b': np.ndarray
      - 'hedge_ratio': np.ndarray
      - 'zscore': np.ndarray
    """
    # Helper to normalize inputs to pd.Series
    def _to_series(x, name: str) -> pd.Series:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError(f"{name} DataFrame must contain 'close' column")
            return pd.Series(x['close'].values)
        if isinstance(x, pd.Series):
            return pd.Series(x.values)
        if isinstance(x, np.ndarray):
            return pd.Series(x)
        # Try to coerce other iterables
        return pd.Series(np.array(x))

    close_a_s = _to_series(asset_a, 'asset_a')
    close_b_s = _to_series(asset_b, 'asset_b')

    close_a = close_a_s.values
    close_b = close_b_s.values

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)

    # Rolling hedge ratio (OLS slope of A ~ B)
    hedge_ratio = np.full(n, np.nan)

    # Ensure lookbacks are sensible
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Require all finite values in the window for stability
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < hedge_lookback:
            # Not enough data in window
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        if x_valid.size < 2 or y_valid.size < 2:
            continue

        # Fully-qualified scipy call as required
        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_valid, y_valid)
        except Exception:
            # Regression failed for this window
            continue

        # Store slope as hedge ratio
        hedge_ratio[i] = slope

    # Spread
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score
    spread_s = pd.Series(spread)
    spread_mean_s = pd.Series.rolling(spread_s, window=zscore_lookback).mean()
    spread_std_s = pd.Series.rolling(spread_s, window=zscore_lookback).std()

    spread_vals = spread_s.values
    mean_vals = spread_mean_s.values
    std_vals = spread_std_s.values

    # Compute z-score safely (avoid division by zero)
    valid = np.isfinite(mean_vals) & np.isfinite(std_vals) & (std_vals != 0)
    zscore = np.where(valid, (spread_vals - mean_vals) / std_vals, np.nan)

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
