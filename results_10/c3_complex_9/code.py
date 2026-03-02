"""
Pairs trading strategy implementation for vectorbt backtests.

Exports:
- compute_spread_indicators(close_a, close_b=None, hedge_lookback=60, zscore_lookback=20) -> dict
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

Notes:
- Rolling OLS hedge ratio with expanding behavior for warmup periods (min periods = 1).
- Z-score uses rolling mean/std with min_periods=1 so no NaNs after warmup.
- Order function is flexible-mode friendly and returns simple tuples (size, size_type, direction).

CRITICAL: This implementation avoids numba usage and returns plain Python tuples.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _to_1d_array(x: Optional[Any], name: str = "x") -> Optional[np.ndarray]:
    """Convert various input types to 1D numpy array.

    Accepts: pd.Series, pd.DataFrame, numpy arrays (1D or 2D).
    If DataFrame or 2D array is provided, returns the first column.
    Returns None if x is None.
    """
    if x is None:
        return None

    # Pandas objects
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return None
        # Prefer a column named 'close' if exists
        if "close" in x.columns:
            ser = x["close"].iloc[:, 0] if isinstance(x["close"], pd.DataFrame) else x["close"]
            return np.asarray(ser).astype(float)
        # Otherwise take the first column
        return np.asarray(x.iloc[:, 0]).astype(float)

    if isinstance(x, pd.Series):
        return np.asarray(x).astype(float)

    # Numpy array
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # Take first column as default
        if arr.shape[1] >= 1:
            return arr[:, 0].astype(float)
        # Fallback: flatten
        return arr.flatten().astype(float)

    # Fallback: try to flatten
    try:
        return np.asarray(x).flatten().astype(float)
    except Exception:
        raise ValueError(f"Unable to convert {name} to 1D numpy array")


def compute_spread_indicators(
    close_a: Any,
    close_b: Optional[Any] = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio, spread and z-score for a pair of assets.

    This function is flexible in input types. It accepts numpy arrays, pandas Series
    or DataFrames. If only `close_a` is provided and it is a 2D object (DataFrame/2D array)
    with >=2 columns, the first two columns are used as asset A and B respectively.

    Args:
        close_a: Prices for asset A (1D array-like) or a 2D container with at least 2 columns.
        close_b: Prices for asset B (1D array-like). Optional if close_a contains both cols.
        hedge_lookback: Lookback window for rolling OLS regression (used as max window;
                        early bars use expanding window to avoid NaNs).
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dictionary with keys:
            - 'hedge_ratio': numpy array of rolling hedge ratios (length = len(close_a)).
            - 'spread': numpy array of spread values (A - hedge_ratio * B).
            - 'zscore': numpy array of z-scores of the spread.
            - 'rolling_mean': rolling mean of spread.
            - 'rolling_std': rolling std of spread.
    """
    # If close_a is a DataFrame with both assets as columns and close_b is None, try to infer
    a_arr = _to_1d_array(close_a, "close_a")
    b_arr = _to_1d_array(close_b, "close_b")

    # If only close_a provided and is 2D-like (we converted to 1D as first column), but close_b is None,
    # try to extract second column from original if it was a DataFrame or 2D-array.
    if b_arr is None:
        # Attempt to extract second column from original close_a (if it was dataframe/2d)
        if isinstance(close_a, pd.DataFrame) and close_a.shape[1] >= 2:
            b_arr = np.asarray(close_a.iloc[:, 1]).astype(float)
        else:
            arr = np.asarray(close_a)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                b_arr = arr[:, 1].astype(float)

    if a_arr is None or b_arr is None:
        raise ValueError("Unable to parse input price series for both assets")

    if a_arr.shape[0] != b_arr.shape[0]:
        # Try to align via pandas if possible
        try:
            a_ser = pd.Series(a_arr)
            b_ser = pd.Series(b_arr)
            joined = pd.concat([a_ser, b_ser], axis=1).dropna()
            a_arr = joined.iloc[:, 0].to_numpy(dtype=float)
            b_arr = joined.iloc[:, 1].to_numpy(dtype=float)
        except Exception:
            raise ValueError("Input series must have the same length")

    # Work with numpy arrays
    a = np.asarray(a_arr, dtype=float)
    b = np.asarray(b_arr, dtype=float)

    n = a.shape[0]
    if n == 0:
        return {
            "hedge_ratio": np.array([]),
            "spread": np.array([]),
            "zscore": np.array([]),
            "rolling_mean": np.array([]),
            "rolling_std": np.array([]),
        }

    hedge_ratio = np.zeros(n, dtype=float)

    # Rolling OLS (A_t ~ beta * B_t + intercept) computed up to and including t
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        ya = a[start : i + 1]
        xb = b[start : i + 1]

        # Drop NaNs in window
        mask = ~np.isnan(ya) & ~np.isnan(xb)
        ya = ya[mask]
        xb = xb[mask]

        if ya.size == 0:
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else 0.0
            continue

        if ya.size >= 2:
            try:
                lr = linregress(xb, ya)
                beta = float(lr.slope)
            except Exception:
                beta = float(ya[-1] / xb[-1]) if xb[-1] != 0 else 0.0
        else:
            beta = float(ya[-1] / xb[-1]) if xb[-1] != 0 else 0.0

        hedge_ratio[i] = beta

    # Compute spread and rolling stats
    spread = a - hedge_ratio * b
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    eps = 1e-8
    zscore = np.zeros(n, dtype=float)
    valid = roll_std > eps
    zscore[valid] = (spread[valid] - roll_mean[valid]) / roll_std[valid]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
        "rolling_mean": roll_mean,
        "rolling_std": roll_std,
    }


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
) -> Tuple[float, int, int]:
    """Order function for flexible multi-asset vectorbt backtest.

    This function is called for each asset (col) at each bar. It returns a tuple
    (size, size_type, direction) or (np.nan, 0, 0) if no order should be placed.

    Position sizing and trade rules:
    - Entry: |zscore| > entry_threshold
        - z > entry_threshold: Short A (1 unit), Long B (hedge_ratio units)
        - z < -entry_threshold: Long A (1 unit), Short B (hedge_ratio units)
      Where 1 unit for A is scaled to `notional_per_leg` (i.e. shares_A = notional_per_leg / price_A)
      and asset B shares = hedge_ratio * shares_A.

    - Exit: z-score crosses 0 -> close both legs
    - Stop-loss: |z-score| > stop_threshold -> close both legs
    """
    # Constants matching vectorbt enums in integer form (commonly: 0=Amount, 1=Long/Buy, 2=Short/Sell)
    SIZE_TYPE_AMOUNT = 0
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Bounds check
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hedge = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0

    if price_a <= 0 or np.isnan(price_a):
        return (np.nan, 0, 0)

    base_shares_a = float(notional_per_leg) / price_a
    target_shares_b = abs(hedge) * base_shares_a

    def close_position(size: float, current_pos: float) -> Tuple[float, int, int]:
        if size <= 0 or abs(current_pos) < 1e-12:
            return (np.nan, 0, 0)
        if current_pos > 0:
            return (abs(size), SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
        else:
            return (abs(size), SIZE_TYPE_AMOUNT, DIRECTION_LONG)

    # Stop-loss: immediate close
    if not np.isnan(z) and abs(z) > stop_threshold:
        return close_position(abs(pos_now), pos_now)

    # Exit on z crossing 0
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan
    if not np.isnan(z_prev) and not np.isnan(z):
        if abs(pos_now) > 1e-12:
            crossed_to_zero = (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold)
            if crossed_to_zero:
                return close_position(abs(pos_now), pos_now)

    # Entry logic - only when flat
    if abs(pos_now) < 1e-12 and not np.isnan(z):
        if z > entry_threshold:
            if col == 0:
                return (abs(base_shares_a), SIZE_TYPE_AMOUNT, DIRECTION_SHORT)
            else:
                return (abs(target_shares_b), SIZE_TYPE_AMOUNT, DIRECTION_LONG)

        if z < -entry_threshold:
            if col == 0:
                return (abs(base_shares_a), SIZE_TYPE_AMOUNT, DIRECTION_LONG)
            else:
                return (abs(target_shares_b), SIZE_TYPE_AMOUNT, DIRECTION_SHORT)

    return (np.nan, 0, 0)
