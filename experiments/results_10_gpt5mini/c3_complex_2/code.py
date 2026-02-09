import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union, Sequence


def compute_spread_indicators(
    close_a: Union[Sequence[float], np.ndarray],
    close_b: Union[Sequence[float], np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a: Prices for asset A (array-like)
        close_b: Prices for asset B (array-like)
        hedge_lookback: Lookback window for rolling OLS regression (uses up to this many past bars).
                        If fewer bars are available, uses all available past bars (no lookahead).
        zscore_lookback: Lookback window for rolling mean/std used to compute z-score.

    Returns:
        Dict containing numpy arrays (same length as inputs) with keys:
            - 'hedge_ratio': rolling OLS slope (beta)
            - 'intercept': rolling OLS intercept
            - 'spread': Price_A - (hedge_ratio * Price_B)
            - 'rolling_mean': rolling mean of spread
            - 'rolling_std': rolling std of spread
            - 'zscore': normalized spread

    Notes:
        - This implementation is careful to avoid lookahead: all rolling computations use data up to
          and including the current bar only.
        - For early bars with insufficient data, expanding windows are used (min(i+1, lookback)).
        - No NaNs are produced after the early warmup period; when std is zero, z-score is set to 0.
    """
    # Convert inputs to numpy arrays of floats
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = len(a)

    hedge_ratio = np.zeros(n, dtype=float)
    intercept = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    rolling_mean = np.zeros(n, dtype=float)
    rolling_std = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Rolling OLS (with intercept) for hedge ratio (beta)
    for i in range(n):
        start = 0 if i - hedge_lookback + 1 < 0 else i - hedge_lookback + 1
        x = b[start : i + 1]
        y = a[start : i + 1]

        # Pairwise valid (in case of NaNs)
        valid = ~np.isnan(x) & ~np.isnan(y)
        xw = x[valid]
        yw = y[valid]

        m = len(xw)
        if m < 2:
            beta = 0.0
            alpha = 0.0
        else:
            mean_x = xw.mean()
            mean_y = yw.mean()
            denom = ((xw - mean_x) ** 2).sum()
            if denom <= 1e-12:
                beta = 0.0
            else:
                beta = ((xw - mean_x) * (yw - mean_y)).sum() / denom
            alpha = mean_y - beta * mean_x

        hedge_ratio[i] = beta
        intercept[i] = alpha

        # Spread per specification: Price_A - (hedge_ratio * Price_B)
        # Use current prices (no intercept term in spread formula by spec)
        if np.isnan(a[i]) or np.isnan(b[i]):
            spread[i] = np.nan
        else:
            spread[i] = a[i] - beta * b[i]

    # Rolling mean/std for z-score
    for i in range(n):
        start = 0 if i - zscore_lookback + 1 < 0 else i - zscore_lookback + 1
        window = spread[start : i + 1]
        window_valid = window[~np.isnan(window)]
        m = len(window_valid)
        if m == 0:
            mu = 0.0
            sigma = 0.0
        elif m == 1:
            mu = float(window_valid[0])
            sigma = 0.0
        else:
            mu = window_valid.mean()
            # population std (ddof=0) - consistent and deterministic
            sigma = float(window_valid.std(ddof=0))

        rolling_mean[i] = mu
        rolling_std[i] = sigma

        if np.isnan(spread[i]):
            zscore[i] = np.nan
        else:
            if sigma <= 1e-12:
                zscore[i] = 0.0
            else:
                zscore[i] = (spread[i] - mu) / sigma

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "zscore": zscore,
    }


def order_func(
    c: Any,
    close_a: Sequence[float],
    close_b: Sequence[float],
    zscore: Sequence[float],
    hedge_ratio: Sequence[float],
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset vectorbt backtest.

    This function is called once per asset (column) per bar by the wrapper. It returns a tuple
    (size, size_type, direction). If size is np.nan, no order is emitted for that call.

    Sizing logic:
      - Fixed notional per leg: notional_per_leg (USD) / price -> units for Asset A
      - Asset B units = hedge_ratio * units_A

    Trading rules (per specification):
      - Entry: zscore > +entry_threshold -> Short A (1 unit), Long B (hedge_ratio units)
               zscore < -entry_threshold -> Long A (1 unit), Short B (hedge_ratio units)
      - Exit: zscore crosses 0.0 -> close both positions
      - Stop-loss: |zscore| > stop_threshold -> close both positions

    Notes:
      - This function uses only information up to current bar (c.i) and previous bar (for cross detection).
      - It expects c to provide attributes: i (int), col (int), and position_now (current position in units).
      - Returned size is in units (absolute), size_type is set to 0 (absolute amount), direction uses
        vectorbt's expected integer mapping: 1 for buy/long, 2 for sell/short.
    """
    # Extract index and column
    i = int(getattr(c, "i"))
    col = int(getattr(c, "col"))

    # Current position for this asset (in units). Flex wrapper provides position_now; real OrderContext exposes
    # different attributes but the wrapper normalizes for tests.
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Arrays may be numpy arrays or lists; ensure numpy indexing
    close_a_arr = np.asarray(close_a, dtype=float)
    close_b_arr = np.asarray(close_b, dtype=float)
    z_arr = np.asarray(zscore, dtype=float)
    hr_arr = np.asarray(hedge_ratio, dtype=float)

    # Safety checks
    if i < 0 or i >= len(z_arr):
        return (np.nan, 0, 0)

    z = float(z_arr[i])
    hr = float(hr_arr[i])

    price_a = float(close_a_arr[i])
    price_b = float(close_b_arr[i])

    # If price is invalid, do nothing
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute unit sizes for Asset A and Asset B (units)
    qty_a = float(notional_per_leg) / price_a
    qty_b = hr * qty_a

    # Determine desired positions (units) for Asset A and Asset B based on zscore
    desired_a: float
    desired_b: float

    # Highest priority: stop-loss
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # Check zero crossing for exit (requires previous bar)
        crossed_zero = False
        if i > 0 and np.isfinite(z_arr[i - 1]):
            prev_z = float(z_arr[i - 1])
            if prev_z * z < 0:
                crossed_zero = True

        if crossed_zero:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # Entries
            if z > entry_threshold:
                # Short A, Long B
                desired_a = -qty_a
                # Asset B position should be scaled by hedge ratio and opposite sign
                desired_b = -hr * desired_a
            elif z < -entry_threshold:
                # Long A, Short B
                desired_a = qty_a
                desired_b = -hr * desired_a
            else:
                # No trading signal - keep existing position
                return (np.nan, 0, 0)

    # Choose which asset this call refers to and compute size to trade (units)
    desired = desired_a if col == 0 else desired_b

    # Compute delta from current position
    size_delta = desired - pos_now

    # If no change required (within tolerance), emit no order
    if abs(size_delta) <= 1e-8:
        return (np.nan, 0, 0)

    # Direction mapping for vectorbt: 1 = buy/long, 2 = sell/short
    direction = 1 if size_delta > 0 else 2
    size = float(abs(size_delta))

    # size_type = 0 -> absolute number of units
    size_type = 0

    return (size, int(size_type), int(direction))
