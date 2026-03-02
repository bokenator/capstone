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
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Guard against invalid prices
    if not np.isfinite(price_a) or price_a <= 0 or not np.isfinite(price_b) or price_b <= 0:
        return (np.nan, 0, 0)

    # Base sizing: number of shares for Asset A based on fixed notional per leg
    shares_a = float(notional_per_leg) / price_a
    # Asset B sized relative to Asset A using hedge ratio (units of B per A)
    shares_b = hr * shares_a

    # Determine previous z for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Stop-loss: if |z| > stop_threshold, close any open position
    if np.isfinite(z) and abs(z) > stop_threshold:
        if pos != 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Exit on crossing zero: close when z crosses 0 (sign change) or hits exactly 0 from a non-zero
    crossed_zero = False
    if not np.isnan(prev_z) and not np.isnan(z):
        if prev_z * z < 0:
            crossed_zero = True
        elif prev_z != 0 and z == 0:
            crossed_zero = True

    if crossed_zero:
        if pos != 0:
            return (-pos, 0, 0)
        return (np.nan, 0, 0)

    # Entry logic
    # If z > entry_threshold: Short A, Long B
    # If z < -entry_threshold: Long A, Short B
    desired_a = None
    desired_b = None

    if z > entry_threshold:
        desired_a = -shares_a
        desired_b = shares_b
    elif z < -entry_threshold:
        desired_a = shares_a
        desired_b = -shares_b

    # If no trading signal, do nothing (keep existing positions)
    if desired_a is None:
        return (np.nan, 0, 0)

    # Determine which asset we're processing and return the difference to desired position
    if col == 0:
        # Asset A side
        # If already at desired size (within tolerance), do nothing
        if np.isclose(pos, desired_a, rtol=1e-6, atol=1e-8):
            return (np.nan, 0, 0)
        # Order to adjust to desired position
        size = desired_a - pos
        return (float(size), 0, 0)
    else:
        # Asset B side
        if np.isclose(pos, desired_b, rtol=1e-6, atol=1e-8):
            return (np.nan, 0, 0)
        size = desired_b - pos
        return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either DataFrames with a 'close' column or plain numpy arrays / Series for compatibility
    with the backtest runner.
    """
    # Extract close arrays from inputs (support multiple input types)
    def _to_close_array(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise ValueError("DataFrame input must contain a 'close' column")
            return x['close'].values.astype(float)
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # Assume array-like
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        return arr.astype(float)

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Rolling hedge ratio using OLS (no lookahead): at time i we use past data up to i (inclusive)
    hedge_ratio = np.full(n, np.nan)

    # To avoid too-early noisy regressions, require at least min_period observations
    min_period = min(hedge_lookback, 20)
    if min_period < 2:
        min_period = 2

    for i in range(n):
        # window length (use up to hedge_lookback or available samples)
        window = min(i + 1, hedge_lookback)
        if window < min_period:
            continue
        start = i - window + 1
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]
        # Ensure finite values and variance in x
        if x.size < 2 or np.isnan(x).any() or np.isnan(y).any() or np.nanstd(x) == 0:
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            # In case regression fails for numerical reasons, skip
            hedge_ratio[i] = np.nan

    # Spread uses hedge ratio at the same index (hedge_ratio computed from past data only)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (no lookahead): rolling window ending at current index
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use population std (ddof=0) for z-score stability
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    # Where std == 0 but mean is finite, set zscore to 0 (no dispersion)
    zero_std_idx = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std == 0)
    zscore[zero_std_idx] = 0.0

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }