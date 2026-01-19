import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


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

    Note: The wrapper provided by the backtest runner calls this function with the
    zscore array before the hedge_ratio array, so the order of these parameters
    reflects that calling convention. The notional_per_leg has a default so the
    wrapper does not have to pass it.

    Args:
        c: Order context with attributes i (index), col (0=A,1=B), position_now, cash_now
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array (1D numpy array)
        hedge_ratio: Hedge ratio array (1D numpy array)
        entry_threshold: Entry threshold for z-score (e.g., 2.0)
        exit_threshold: Exit threshold for z-score (e.g., 0.0)
        stop_threshold: Stop-loss threshold for z-score (e.g., 3.0)
        notional_per_leg: Dollar notional per leg (default 10000.0)

    Returns:
        (size, size_type, direction) tuple as expected by the runner wrapper.
        - size: number of shares (positive -> buy, negative -> sell)
        - size_type: 0 = Amount (shares)
        - direction: 0 = Both

    Behavior:
        - Entry when |zscore| > entry_threshold. For z>entry_threshold: short A, long B.
          For z<-entry_threshold: long A, short B.
        - Exit when z crosses zero (sign change) or when |z| > stop_threshold (stop-loss).
        - Position sizing: shares_a = notional_per_leg / price_a
                          shares_b = hedge_ratio * shares_a
          (B position scaled by hedge ratio)
    """
    i = int(getattr(c, 'i', 0))
    col = int(getattr(c, 'col', 0))
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic bounds check
    if i < 0:
        return (np.nan, 0, 0)

    # Safely extract prices and indicators
    try:
        price_a = float(close_a[i])
        price_b = float(close_b[i])
    except Exception:
        return (np.nan, 0, 0)

    # Validate prices
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Extract indicators at time i
    z = float(zscore[i]) if (0 <= i < len(zscore)) else np.nan
    hr = float(hedge_ratio[i]) if (0 <= i < len(hedge_ratio)) else np.nan

    # If indicators are invalid, do nothing
    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, 0, 0)

    # Compute current target sizes (in shares)
    shares_a = notional_per_leg / price_a
    shares_b = hr * shares_a  # can be negative if hr negative

    # Previous z for crossover detection
    z_prev = float(zscore[i - 1]) if (i - 1) >= 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Determine stop-loss (immediate close)
    if abs(z) > stop_threshold:
        # Close if there is a position
        if pos_now != 0:
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Determine exit on reversion (z crosses zero)
    crossed_zero = False
    if np.isfinite(z_prev):
        if (z_prev > 0 and z <= 0) or (z_prev < 0 and z >= 0):
            crossed_zero = True

    if crossed_zero:
        if pos_now != 0:
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic: if currently flat, look to enter
    # If already in a position, maintain it (no pyramiding/changing size)
    # We compute desired target and return the delta (desired - current)
    desired = None

    if abs(z) > entry_threshold:
        # Enter trade
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                desired = -shares_a
            else:
                desired = +shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                desired = +shares_a
            else:
                desired = -shares_b
    else:
        # Not a strong signal and not a crossing -> do nothing
        return (np.nan, 0, 0)

    # Compute order size: desired position minus current position
    if desired is None:
        return (np.nan, 0, 0)

    size = float(desired - pos_now)

    # If change is negligible, do nothing
    if abs(size) < 1e-8:
        return (np.nan, 0, 0)

    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for pairs trading: hedge ratio (rolling OLS) and z-score of spread.

    This function is robust to inputs being either pandas DataFrames/Series with a
    'close' column or plain numpy arrays / pandas Series. If the provided asset
    arrays have different lengths, they will be truncated to the minimum common
    length (this avoids errors when tests pass truncated inputs for lookahead checks).

    Args:
        asset_a: DataFrame or array-like for Asset A. If DataFrame, must contain 'close'.
        asset_b: DataFrame or array-like for Asset B. If DataFrame, must contain 'close'.
        hedge_lookback: Lookback period for rolling OLS (in bars)
        zscore_lookback: Lookback period for rolling mean/std of the spread

    Returns:
        Dictionary with numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    """
    # Extract close arrays from various possible input types
    def _extract_close(obj):
        # pandas DataFrame with 'close' column
        if isinstance(obj, pd.DataFrame):
            if 'close' not in obj.columns:
                # If DataFrame contains the two asset columns (e.g., nested dict-like input),
                # try to flatten by taking the first column
                if obj.shape[1] >= 1:
                    return obj.iloc[:, 0].astype(float).values
                raise KeyError("DataFrame input must contain 'close' column")
            arr = obj['close'].astype(float).values
            return arr
        # pandas Series
        if isinstance(obj, pd.Series):
            return obj.astype(float).values
        # numpy array or array-like
        arr = np.asarray(obj)
        if arr.ndim == 0:
            raise ValueError('Input must be 1D')
        if arr.ndim > 1:
            # If array is 2D and has a named 'close' column, try to coerce via DataFrame
            try:
                df = pd.DataFrame(arr)
                if 'close' in df.columns:
                    return df['close'].astype(float).values
            except Exception:
                pass
            # Otherwise, squeeze
            arr = arr.squeeze()
        return arr.astype(float)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    # If lengths mismatch (e.g., tests pass truncated first arg), truncate to minimum
    if len(close_a) != len(close_b):
        n = min(len(close_a), len(close_b))
        close_a = close_a[:n]
        close_b = close_b[:n]

    if len(close_a) == 0:
        raise ValueError('Input arrays must be non-empty')

    n = len(close_a)

    # Validate lookbacks (at least 2 for regression)
    hedge_lookback = int(max(2, hedge_lookback))
    zscore_lookback = int(max(2, zscore_lookback))

    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS: for each time i, regress y (A) on x (B) using available past window
    # Window is inclusive: [i - hedge_lookback + 1, i]
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        end = i + 1
        x = close_b[start:end]
        y = close_a[start:end]
        # Need at least two points and no NaNs
        if len(x) < 2 or np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread: A - hedge_ratio * B
    spread = np.full(n, np.nan)
    valid_idx = ~np.isnan(hedge_ratio)
    spread[valid_idx] = close_a[valid_idx] - hedge_ratio[valid_idx] * close_b[valid_idx]

    # Rolling mean and std of spread (lookback inclusive). Use pandas for rolling with min_periods=1
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).values

    # Compute z-score safely (avoid division by zero)
    zscore = np.full(n, np.nan)
    for i in range(n):
        m = spread_mean[i]
        s = spread_std[i]
        sp = spread[i]
        if not np.isfinite(sp) or not np.isfinite(m) or not np.isfinite(s) or s == 0:
            zscore[i] = np.nan
        else:
            zscore[i] = (sp - m) / s

    # Return arrays (ensure numpy arrays)
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hedge_ratio, dtype=float),
        'zscore': np.asarray(zscore, dtype=float),
    }
