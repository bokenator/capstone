import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def _to_1d_array(x: Any) -> np.ndarray:
    """Convert input to a 1D numpy array of floats.

    Handles numpy arrays, pandas Series/DataFrame and list-like objects.
    If a DataFrame with multiple columns is passed, the first column is used.
    """
    if isinstance(x, pd.Series):
        return x.values.astype(float)
    if isinstance(x, pd.DataFrame):
        # Prefer 'close' column if present
        if "close" in x.columns:
            return x["close"].values.astype(float)
        # If single column, use it
        if x.shape[1] == 1:
            return x.iloc[:, 0].values.astype(float)
        # Fallback: use the first column
        return x.iloc[:, 0].values.astype(float)

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        # If 2D with one column, squeeze
        if arr.shape[1] == 1:
            return arr[:, 0]
        # Otherwise, take the first column
        return arr[:, 0]
    return arr


def compute_spread_indicators(
    close_a: Any,
    close_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of asset price series.

    Args:
        close_a: Prices for asset A (1D array-like, Series, or DataFrame with a 'close' column).
        close_b: Prices for asset B (1D array-like, Series, or DataFrame with a 'close' column).
        hedge_lookback: Lookback window for rolling OLS (uses up to i+1 points when i+1 < lookback).
        zscore_lookback: Lookback window for rolling mean/std of spread (uses up to i+1 points when i+1 < lookback).

    Returns:
        Dict containing numpy arrays: 'zscore' and 'hedge_ratio' (and 'spread').

    Notes:
        - This implementation is causal: computations at time i use only data up to and including i.
        - For early bars where a full lookback is not available, a shorter window is used (expanding window).
        - If variance in the regression x window is zero, the hedge ratio is set to 0.0.
        - Z-score uses population std (ddof=0). If std == 0, z-score is set to 0.0 when spread available.
    """
    ca = _to_1d_array(close_a)
    cb = _to_1d_array(close_b)

    if ca.shape[0] != cb.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = ca.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Rolling computations (causal)
    for i in range(n):
        # Rolling OLS for hedge ratio: regress A on B using data up to i, with window capped by hedge_lookback
        lb = min(hedge_lookback, i + 1)
        start = i - lb + 1
        x = cb[start : i + 1]
        y = ca[start : i + 1]

        # Require at least 2 points to compute slope; otherwise try to propagate previous hedge ratio
        if lb >= 2 and not (np.isnan(x).any() or np.isnan(y).any()):
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                slope = 0.0
            else:
                slope = np.sum((x - x_mean) * (y - y_mean)) / denom
            hedge_ratio[i] = slope
        else:
            # Propagate previous hedge ratio if available (causal), otherwise leave as NaN
            hedge_ratio[i] = hedge_ratio[i - 1] if i > 0 else np.nan

        # Compute spread at i if hedge ratio is available
        if not np.isnan(hedge_ratio[i]):
            spread[i] = ca[i] - hedge_ratio[i] * cb[i]
        else:
            spread[i] = np.nan

        # Rolling mean/std for spread to compute z-score (causal, window capped by zscore_lookback)
        lb2 = min(zscore_lookback, i + 1)
        start2 = i - lb2 + 1
        window = spread[start2 : i + 1]

        # If all values in window are NaN, zscore remains NaN
        if np.isnan(window).all():
            zscore[i] = np.nan
        else:
            mean = np.nanmean(window)
            std = np.nanstd(window)
            if std == 0 or np.isnan(std):
                # If std is zero, set zscore to 0 when spread is available
                zscore[i] = 0.0 if not np.isnan(spread[i]) else np.nan
            else:
                zscore[i] = (spread[i] - mean) / std

    return {"zscore": zscore, "hedge_ratio": hedge_ratio, "spread": spread}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading (flexible multi-asset mode).

    The function is called once per asset (column) per bar. It must return a tuple:
        (size, size_type, direction)
    where size is the absolute size (number of units) to trade, size_type is an integer
    that tells vectorbt how to interpret `size` (0 = units), and direction is an integer
    indicating buy/long vs sell/short (1 = buy/long, 2 = sell/short). To signal no order,
    return a tuple with size = np.nan.

    Logic implemented:
        - Use causal zscore and hedge_ratio at index c.i.
        - Entry: zscore > entry_threshold -> short A, long B
                 zscore < -entry_threshold -> long A, short B
                 Sizes: base units for A = notional_per_leg / price_a; B units = hedge_ratio * units_a
        - Exit: zscore crosses zero (sign change) OR abs(zscore) <= exit_threshold -> close both legs
        - Stop-loss: abs(zscore) > stop_threshold -> close both legs
        - If no action is required for this asset, return (np.nan, 0, 0) to indicate NoOrder.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely extract prices and indicators at time i
    try:
        price_a = float(close_a[i])
        price_b = float(close_b[i])
    except Exception:
        return (np.nan, 0, 0)

    # If indicators are not available at this bar, do nothing
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan

    if np.isnan(z) or np.isnan(hr) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Base unit sizing for asset A, and apply hedge ratio for B units so that:
    # units_b = hedge_ratio * units_a
    units_a = float(notional_per_leg) / price_a
    units_b = hr * units_a

    # Get current position for this asset
    current_pos = float(getattr(c, "position_now", 0.0))

    # Determine target positions for both assets
    target_a = 0.0
    target_b = 0.0

    # Priority: stop-loss (close), then normal exit (crossing zero / close), then entry
    if abs(z) > stop_threshold:
        # Stop loss: close both legs
        target_a = 0.0
        target_b = 0.0
    else:
        # Check for zero crossing (use previous z if available)
        crossed_zero = False
        if i > 0 and not np.isnan(zscore[i - 1]):
            zprev = float(zscore[i - 1])
            if zprev * z < 0:
                crossed_zero = True

        if crossed_zero or abs(z) <= exit_threshold:
            # Exit on mean reversion
            target_a = 0.0
            target_b = 0.0
        elif z > entry_threshold:
            # Short A, Long B
            target_a = -units_a
            target_b = +units_b
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +units_a
            target_b = -units_b
        else:
            # Within thresholds: no action
            return (np.nan, 0, 0)

    # Select the target for the current column and compute required trade size (delta)
    target = target_a if col == 0 else target_b
    delta = target - current_pos

    # If delta is effectively zero, do not place an order
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    size = float(abs(delta))
    size_type = 0  # interpret `size` as number of units
    # Direction: 1 = buy/long, 2 = sell/short
    direction = 1 if delta > 0 else 2

    return (size, int(size_type), int(direction))


# Expose only the required symbols
__all__ = ["compute_spread_indicators", "order_func"]
