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

    This flexible order function uses value-based sizing (size_type=1) for
    entries so that each leg targets a fixed notional exposure. Exits close
    the current position using amount-based sizing (size_type=0) by returning
    -position_now.

    Args:
        c: vectorbt OrderContext-like object with attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float) (optional)
        close_a: Close prices for Asset A (1D numpy array)
        close_b: Close prices for Asset B (1D numpy array)
        zscore: Z-score array for the spread (1D numpy array)
        hedge_ratio: Rolling hedge ratio array (1D numpy array)
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        A tuple of (size, size_type, direction) as described in the prompt.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos = float(getattr(c, "position_now", 0.0))

    # Validate index
    if i < 0 or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    # Read zscore for current bar
    z = zscore[i] if (zscore is not None and i < len(zscore)) else np.nan
    if not np.isfinite(z):
        return (np.nan, 0, 0)

    # Prices
    price = float(close_a[i]) if col == 0 else float(close_b[i])
    if not np.isfinite(price) or price <= 0.0:
        return (np.nan, 0, 0)

    # Previous zscore for crossing detection
    prev_z = zscore[i - 1] if i > 0 and i - 1 < len(zscore) else np.nan

    # STOP-LOSS: if |z| > stop_threshold, close any existing position
    if np.isfinite(z) and np.abs(z) > stop_threshold:
        if pos != 0.0:
            # Close full position (amount = -current position)
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # EXIT: z-score crosses the exit_threshold (typically 0.0)
    if np.isfinite(prev_z) and np.isfinite(z):
        crossed_to_exit = (
            (prev_z > exit_threshold and z <= exit_threshold) or
            (prev_z < exit_threshold and z >= exit_threshold)
        )
        if crossed_to_exit:
            if pos != 0.0:
                return (-pos, 0, 0)
            return (np.nan, 0, 0)

    # ENTRY: only enter if currently flat for this asset
    if pos == 0.0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A by fixed notional (value order)
                return (-float(notional_per_leg), 1, 0)
            else:
                # Long Asset B by fixed notional
                return (float(notional_per_leg), 1, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                return (float(notional_per_leg), 1, 0)
            else:
                return (-float(notional_per_leg), 1, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict:
    """
    Precompute indicators for the pairs strategy.

    Accepts either pandas DataFrames/Series with a 'close' column or 1D numpy
    arrays for prices. Returns a dict containing 'close_a', 'close_b',
    'hedge_ratio', and 'zscore' as numpy arrays of the same length.

    Hedge ratio is computed with a rolling OLS (slope only) using
    scipy.stats.linregress on windows of length `hedge_lookback`.

    Z-score uses a rolling mean/std of the spread over `zscore_lookback`.
    """
    def _extract_close(x):
        # Accept pd.DataFrame with 'close', pd.Series, or numpy array-like
        if isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise ValueError("DataFrame input must contain 'close' column")
            return x["close"].values.astype(float)
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # Assume array-like
        arr = np.array(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Price input must be 1D array-like of closes")
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)

    # Initialize hedge ratio with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS (slope) for hedge ratio
    if hedge_lookback >= 1 and n >= hedge_lookback:
        for i in range(hedge_lookback, n):
            y = close_a[i - hedge_lookback:i]
            x = close_b[i - hedge_lookback:i]
            mask = np.isfinite(x) & np.isfinite(y)
            # Require a full window without NaNs for regression
            if np.sum(mask) != hedge_lookback:
                continue
            # Avoid degenerate x series (zero variance)
            if np.std(x[mask]) == 0.0:
                continue
            # Compute slope using fully-qualified scipy.stats.linregress
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
            # Only accept finite slopes
            if np.isfinite(slope):
                hedge_ratio[i] = float(slope)

    # Compute spread using the rolling hedge ratio (aligned by index)
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = np.isfinite(hedge_ratio) & np.isfinite(close_a) & np.isfinite(close_b)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std of spread
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    spread_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    mask_z = np.isfinite(spread) & np.isfinite(spread_mean) & np.isfinite(spread_std) & (spread_std > 0.0)
    zscore[mask_z] = (spread[mask_z] - spread_mean[mask_z]) / spread_std[mask_z]

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
