# Updated implementation for pairs trading indicators and order function
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert various array-like inputs to a 1D numpy array (ravel if necessary).

    This helper does NOT use any future data and only reshapes the provided input.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    if isinstance(x, pd.DataFrame):
        # Caller should handle DataFrame with multiple columns; ravel for single-col df
        arr = x.to_numpy(dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr
    arr = np.asarray(x, dtype=float)
    return arr


def compute_spread_indicators(
    close_a: Any,
    close_b: Optional[Any] = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs trading strategy.

    Accepts flexible input types: numpy arrays, pandas Series/DataFrame. If the caller
    provides a single 2-column DataFrame/ndarray as the first argument and omits the
    second argument, the two columns are treated as asset A and asset B respectively.

    Args:
        close_a: Prices for asset A (or a 2-column container with [A, B])
        close_b: Prices for asset B (optional if close_a already contains both series)
        hedge_lookback: Lookback window (in bars) for rolling OLS to compute hedge ratio.
                        Uses an expanding window for the initial bars (min_periods=1) so
                        that no excessive NaNs are produced.
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        Dictionary with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (same length as inputs)
            - 'zscore': np.ndarray of z-score of the spread (same length as inputs)

    The implementation avoids lookahead: all rolling statistics use only data up to
    and including the current bar.
    """
    # Flexible parsing of inputs
    a_raw = close_a
    b_raw = close_b

    # If first argument is a DataFrame or 2D array and second argument not provided,
    # treat it as two-column input
    if b_raw is None:
        if isinstance(a_raw, pd.DataFrame):
            if a_raw.shape[1] >= 2:
                a = a_raw.iloc[:, 0].to_numpy(dtype=float)
                b = a_raw.iloc[:, 1].to_numpy(dtype=float)
            else:
                # Single-column DataFrame but no second argument
                raise ValueError("close_b must be provided when close_a has a single column")
        else:
            a_arr = _to_1d_array(a_raw)
            if a_arr.ndim == 2 and a_arr.shape[1] >= 2:
                # Treat as two columns: first two columns are A and B
                a = a_arr[:, 0]
                b = a_arr[:, 1]
            else:
                raise ValueError("close_b must be provided when close_a does not contain two columns")
    else:
        # Both arguments provided - convert to 1D arrays where possible
        a_conv = _to_1d_array(a_raw)
        b_conv = _to_1d_array(b_raw)

        # If either conversion returned a 2D array (e.g. DataFrame with 2 cols), try to interpret
        if a_conv.ndim == 2 and a_conv.shape[1] >= 2 and (b_conv is None or b_conv.ndim != 1):
            a = a_conv[:, 0]
            b = a_conv[:, 1]
        else:
            # Ravel to 1D if necessary
            a = a_conv.ravel()
            b = b_conv.ravel()

    # Final validation
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays after parsing")
    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    n = len(a)

    # Compute rolling hedge ratio (OLS slope) using only past data per index
    hedge_ratio = np.empty(n, dtype=float)
    prev_slope = 1.0
    for i in range(n):
        start = 0 if i - hedge_lookback + 1 < 0 else i - hedge_lookback + 1
        y = a[start : i + 1]
        x = b[start : i + 1]

        if x.size >= 2:
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom <= 1e-12:
                slope = prev_slope
            else:
                slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        else:
            slope = prev_slope

        hedge_ratio[i] = float(slope)
        prev_slope = slope

    spread = a - hedge_ratio * b

    # Rolling mean/std for z-score with min_periods=1 and ddof=0
    spread_s = pd.Series(spread)
    roll_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    roll_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    eps = 1e-12
    safe_std = np.where(np.abs(roll_std) < eps, np.nan, roll_std)
    zscore = (spread - roll_mean) / safe_std
    zscore = np.where(np.isnan(zscore), 0.0, zscore)

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10000.0,
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset mode.

    Returns a tuple (size, size_type, direction). Uses TargetValue sizing so that
    `size` is interpreted as a target USD exposure for the asset.

    Direction codes (integers):
        0 -> LongOnly
        1 -> ShortOnly
        2 -> Both

    Size type used:
        4 -> TargetValue
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    cur_z = float(zscore[i]) if not np.isnan(zscore[i]) else 0.0
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else 0.0

    SIZE_TYPE_TARGET_VALUE = 4
    DIRECTION_LONG = 0
    DIRECTION_SHORT = 1
    DIRECTION_BOTH = 2

    # Stop-loss
    if abs(cur_z) > stop_threshold:
        return (0.0, SIZE_TYPE_TARGET_VALUE, DIRECTION_BOTH)

    # Exit on crossing zero (mean reversion)
    crossed_to_zero = False
    if i > 0:
        if prev_z > exit_threshold and cur_z <= exit_threshold:
            crossed_to_zero = True
        if prev_z < exit_threshold and cur_z >= exit_threshold:
            crossed_to_zero = True

    if crossed_to_zero:
        return (0.0, SIZE_TYPE_TARGET_VALUE, DIRECTION_BOTH)

    # Entries
    if cur_z > entry_threshold:
        # Short A, Long B
        if col == 0:
            return (float(notional_per_leg), SIZE_TYPE_TARGET_VALUE, DIRECTION_SHORT)
        else:
            return (float(notional_per_leg), SIZE_TYPE_TARGET_VALUE, DIRECTION_LONG)

    if cur_z < -entry_threshold:
        # Long A, Short B
        if col == 0:
            return (float(notional_per_leg), SIZE_TYPE_TARGET_VALUE, DIRECTION_LONG)
        else:
            return (float(notional_per_leg), SIZE_TYPE_TARGET_VALUE, DIRECTION_SHORT)

    return (np.nan, 0, 0)
