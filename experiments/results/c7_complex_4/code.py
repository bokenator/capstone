from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy


def order_func(
    c: Any,
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
    Generate orders for pairs trading in flexible multi-asset mode.

    Note: The wrapper used by the backtester calls this function with the
    positional order: (c, close_a, close_b, zscore, hedge_ratio, entry_threshold,
    exit_threshold, stop_threshold). We default `notional_per_leg` so it does not
    need to be passed by the wrapper.

    Returns a tuple (size, size_type, direction) where size is number of shares
    (positive = buy, negative = sell), size_type=0 (Amount), direction=0 (Both).
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic bounds checks
    if i < 0:
        return (np.nan, 0, 0)

    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    # Read current indicators/prices
    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    h = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    # If indicators or prices are not available, attempt to safely close any open position
    if np.isnan(z) or np.isnan(h) or np.isnan(price_a) or np.isnan(price_b):
        if pos_now != 0.0:
            # Close existing position to be safe
            return (-pos_now, 0, 0)
        return (np.nan, 0, 0)

    # Determine target sizes in shares
    # Shares for Asset A based on fixed notional
    shares_a = float(notional_per_leg / price_a) if price_a != 0 else 0.0
    # Asset B is scaled by hedge ratio (position_b ~= hedge_ratio * position_a)
    shares_b = float(shares_a * h)

    desired_pos = pos_now  # default: hold

    # Stop-loss: if |z| > stop_threshold -> close both legs
    if abs(z) > stop_threshold:
        desired_pos = 0.0
    else:
        # Entry signals
        if z > entry_threshold:
            # Short A, Long B
            if col == 0:
                desired_pos = -shares_a
            else:
                desired_pos = shares_b
        elif z < -entry_threshold:
            # Long A, Short B
            if col == 0:
                desired_pos = shares_a
            else:
                desired_pos = -shares_b
        else:
            # No new entry. Check for exit on mean reversion (z crossing 0)
            # Only act if we are currently in a trade
            if pos_now != 0.0:
                prev_z = zscore[i - 1] if i > 0 else np.nan
                if np.isfinite(prev_z):
                    # Crossing detection without using np.sign (not in VAS)
                    crossed = False
                    if prev_z * z < 0:
                        crossed = True
                    elif prev_z == 0.0 and z != 0.0:
                        crossed = True
                    elif z == 0.0 and prev_z != 0.0:
                        crossed = True

                    if crossed:
                        desired_pos = 0.0
                    else:
                        return (np.nan, 0, 0)
                else:
                    return (np.nan, 0, 0)
            else:
                return (np.nan, 0, 0)

    # Compute order size (difference between desired and current position)
    size = float(desired_pos - pos_now)

    # If size is effectively zero, do nothing
    if abs(size) < 1e-12:
        return (np.nan, 0, 0)

    # Return as number of shares (Amount)
    return (size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    The hedge ratio at time t is computed using an OLS regression on a rolling
    window that ends at t (inclusive). This ensures no lookahead bias: only
    data up to and including t is used to compute indicators for t.

    The function accepts pandas DataFrame/Series or numpy arrays. If both inputs
    are pandas objects, they are aligned on their index (intersection) so that
    truncated inputs produce consistent, causal outputs.

    Returns a dict with numpy arrays: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.
    """
    # Helper to convert inputs to pd.Series while preserving index when present
    def _to_series_with_index(x) -> pd.Series:
        if isinstance(x, pd.DataFrame):
            if "close" in x.columns:
                return pd.Series(x["close"].values, index=x.index)
            # fallback to first column
            return pd.Series(x.iloc[:, 0].values, index=x.index)
        if isinstance(x, pd.Series):
            return pd.Series(x.values, index=x.index)
        if isinstance(x, np.ndarray):
            return pd.Series(x)
        # fallback for lists/other iterables
        return pd.Series(np.array(x))

    s_a = _to_series_with_index(asset_a)
    s_b = _to_series_with_index(asset_b)

    # If either input had an index (pandas), align to the common index to avoid
    # mismatched lengths when truncated data is provided.
    if hasattr(s_a, "index") and hasattr(s_b, "index"):
        common_index = s_a.index.intersection(s_b.index)
        s_a = s_a.reindex(common_index)
        s_b = s_b.reindex(common_index)

    if len(s_a) != len(s_b):
        # As a final fallback (both numpy-like but lengths differ) raise informative error
        raise ValueError("asset_a and asset_b must have the same length after alignment")

    n = len(s_a)
    a_vals = s_a.values.astype(float)
    b_vals = s_b.values.astype(float)

    # Rolling hedge ratio (OLS). Use a window that ends at i (inclusive) and has
    # size up to hedge_lookback. This is causal and consistent under truncation.
    hedge_ratio = np.full(n, np.nan)

    for i in range(n):
        # window size: at most hedge_lookback, but no more than available points (i+1)
        window = hedge_lookback if (i + 1) >= hedge_lookback else (i + 1)
        if window < 2:
            # Need at least two points for linear regression
            continue
        start = i - window + 1
        x = b_vals[start : i + 1]
        y = a_vals[start : i + 1]

        # Filter non-finite values
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue

        # OLS regression using scipy.stats.linregress
        res = scipy.stats.linregress(x[mask], y[mask])
        slope = res.slope if hasattr(res, "slope") else res[0]
        hedge_ratio[i] = float(slope)

    # Compute spread using the hedge ratio at the same time index (no future data)
    spread = a_vals - hedge_ratio * b_vals

    # Rolling mean/std for z-score. Use min_periods=1 so indicator is available as soon
    # as there is any data (still causal). Use ddof=0 to avoid NaNs for single-sample std.
    spread_series = pd.Series(spread)
    spread_mean = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).mean().values
    try:
        spread_std = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std(ddof=0).values
    except TypeError:
        spread_std = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    for i in range(n):
        s = spread[i]
        m = spread_mean[i]
        sd = spread_std[i]
        if not np.isfinite(s) or not np.isfinite(m) or not np.isfinite(sd):
            continue
        if sd == 0.0:
            zscore[i] = 0.0
        else:
            zscore[i] = (s - m) / sd

    return {
        "close_a": a_vals,
        "close_b": b_vals,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
