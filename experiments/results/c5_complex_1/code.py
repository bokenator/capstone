import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Union


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
    Generate orders for pairs trading. Called by vectorbt's from_order_func (flexible mode).

    Args:
        c: Order context with attributes i (int), col (int), position_now (float), cash_now (float)
        close_a: Close prices for Asset A (np.ndarray)
        close_b: Close prices for Asset B (np.ndarray)
        zscore: Z-score array (np.ndarray)
        hedge_ratio: Hedge ratio array (np.ndarray)
        entry_threshold: Entry z-score threshold
        exit_threshold: Exit threshold (crossing this value triggers exit)
        stop_threshold: Stop-loss threshold (absolute z-score beyond this triggers exit)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        (size, size_type, direction)
    """
    i = int(c.i)
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos = float(getattr(c, "position_now", 0.0))

    # Basic bounds check
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Read indicators at current bar
    # Wrapper may pass arrays in either order; we assume wrapper passes (close_a, close_b, zscore, hedge_ratio, ...)
    z = float(zscore[i]) if i < len(zscore) else np.nan
    h = float(hedge_ratio[i]) if i < len(hedge_ratio) else np.nan

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # If indicators or prices are not available, do nothing
    if np.isnan(z) or np.isnan(h) or np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine target position sizes (in shares) based on fixed notional per leg
    shares_a = float(notional_per_leg) / price_a

    # Determine sign of z to derive desired A position: short A when z>0, long A when z<0
    if z > 0:
        sign_z = 1
    elif z < 0:
        sign_z = -1
    else:
        sign_z = 0

    # Target positions
    pos_a_target = -sign_z * shares_a
    pos_b_target = -h * pos_a_target

    # Priority 1: Stop-loss
    if abs(z) > stop_threshold:
        if pos != 0.0:
            return (-pos, 0, 0)  # close
        return (np.nan, 0, 0)

    # Priority 2: Exit on crossing exit_threshold (usually 0.0)
    if i >= 1:
        prev_z = zscore[i - 1]
        if not np.isnan(prev_z):
            crossed = ((prev_z > exit_threshold and z < exit_threshold) or
                       (prev_z < exit_threshold and z > exit_threshold))
            if crossed:
                if pos != 0.0:
                    return (-pos, 0, 0)
                return (np.nan, 0, 0)

    # Entry logic: enter when |z| > entry_threshold
    if abs(z) > entry_threshold and sign_z != 0:
        target = pos_a_target if col == 0 else pos_b_target
        # If already at target (within tolerance), do nothing
        if np.isclose(pos, target, atol=1e-8):
            return (np.nan, 0, 0)
        # Place order to move from current position to target
        order_size = float(target - pos)
        return (order_size, 0, 0)

    # Otherwise, no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required for the pairs strategy.

    This function is flexible in accepted inputs:
    - asset_a and asset_b can be numpy arrays, pd.Series, or pd.DataFrame with 'close' column
    - asset_a can also be a container (dict or DataFrame) containing both 'asset_a' and 'asset_b' entries

    Returns dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
    Each is a numpy array of the same length.
    """

    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray, dict]) -> np.ndarray:
        # If x is a DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if "close" in x.columns:
                return x["close"].astype(float).values
            # If DataFrame contains numeric column directly (e.g., a single series), try to pick the first numeric column
            numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 1:
                return x[numeric_cols[0]].astype(float).values
            # If it has columns 'asset_a'/'asset_b', handle outside
        if isinstance(x, pd.Series):
            return x.astype(float).values
        if isinstance(x, np.ndarray):
            return x.astype(float)
        if isinstance(x, dict):
            # If dict contains close or nested structure
            if "close" in x:
                return _extract_close(x["close"]) if not isinstance(x["close"], np.ndarray) else np.asarray(x["close"], dtype=float)
            # If dict is nested asset container, let caller handle
        # Fallback: try to coerce to numpy array
        try:
            return np.asarray(x, dtype=float)
        except Exception:
            raise ValueError("Unsupported input type for close extraction")

    # If first argument is a container holding both assets, unpack it
    if isinstance(asset_a, dict) and ("asset_a" in asset_a and "asset_b" in asset_a):
        close_a = _extract_close(asset_a["asset_a"])
        close_b = _extract_close(asset_a["asset_b"])
    elif isinstance(asset_a, pd.DataFrame) and "asset_a" in asset_a.columns and "asset_b" in asset_a.columns:
        # DataFrame with two columns named asset_a and asset_b
        # Columns may be numeric series (close prices)
        close_a = _extract_close(asset_a["asset_a"])  # may succeed even if these are Series
        close_b = _extract_close(asset_a["asset_b"])
    else:
        # Default: extract closes from the two provided objects
        close_a = _extract_close(asset_a)
        close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        # Try to align lengths if one of the inputs was a DataFrame with datetime index
        # If lengths differ, but one of them is longer and contains the other as a prefix, try to truncate the longer to the shorter
        if len(close_a) > len(close_b) and np.allclose(close_a[: len(close_b)], close_a[: len(close_b)], equal_nan=True):
            close_a = close_a[: len(close_b)]
        elif len(close_b) > len(close_a) and np.allclose(close_b[: len(close_a)], close_b[: len(close_a)], equal_nan=True):
            close_b = close_b[: len(close_a)]
        else:
            # As a last resort, attempt to align by intersection of indices if inputs were DataFrames with index
            # If we cannot reconcile lengths, raise error
            raise ValueError("Input arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS hedge ratio using past data (exclude current bar)
    for i in range(n):
        start = max(0, i - hedge_lookback)
        end = i
        if end - start >= 2:
            x = close_b[start:end]
            y = close_a[start:end]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    spread_series = pd.Series(spread)
    # Rolling mean/std that end at current bar (no lookahead)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    zscore = np.full(n, np.nan, dtype=float)
    valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std)) & (spread_std > 0)
    zscore[valid] = (spread[valid] - spread_mean[valid]) / spread_std[valid]

    zero_std_idx = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (spread_std == 0)
    zscore[zero_std_idx] = 0.0

    return {
        "close_a": np.asarray(close_a, dtype=float),
        "close_b": np.asarray(close_b, dtype=float),
        "hedge_ratio": np.asarray(hedge_ratio, dtype=float),
        "zscore": np.asarray(zscore, dtype=float),
    }
