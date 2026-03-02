"""
Pairs trading: compute_spread_indicators and order_func implementation

- compute_spread_indicators: rolling OLS (lookback) hedge ratio, spread, z-score
- order_func: flexible order function for vectorbt.from_order_func (flexible=True)

Notes:
- No Numba usage
- size_type uses 0 (absolute units)
- direction uses 1 for BUY/LONG, 2 for SELL/SHORT
- Position sizing: scale so that Asset A notional == notional_per_leg; Asset B units = hedge_ratio * units_A

"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS of A on B), spread, and z-score.

    Args:
        close_a: 1D array of asset A close prices
        close_b: 1D array of asset B close prices
        hedge_lookback: lookback periods for rolling OLS (slope)
        zscore_lookback: lookback periods for rolling mean/std of spread

    Returns:
        dict with keys:
            - "hedge_ratio": np.ndarray of rolling hedge ratios (slope)
            - "spread": np.ndarray of spread values (A - hedge_ratio * B)
            - "zscore": np.ndarray of z-score of the spread

    Notes:
        - Windows containing any NaNs produce NaN hedge_ratio for that timestamp
        - Uses population std (ddof=0) for z-score calculation
    """
    # Validate and prepare inputs
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)

    if close_a.ndim != 1 or close_b.ndim != 1:
        raise ValueError("close_a and close_b must be 1D arrays")
    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = close_a.shape[0]
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Initialize hedge_ratio with NaNs
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression (A ~ beta * B + intercept) - store beta (slope)
    for i in range(hedge_lookback - 1, n):
        start = i - hedge_lookback + 1
        window_a = close_a[start : i + 1]
        window_b = close_b[start : i + 1]

        # Skip if any NaNs in window
        if np.isnan(window_a).any() or np.isnan(window_b).any():
            continue

        # If regressor (B) has zero variance, slope is undefined
        if np.all(window_b == window_b[0]):
            continue

        try:
            slope, intercept, r_value, p_value, std_err = linregress(window_b, window_a)
        except Exception:
            # In case linregress fails for some numerical reason
            continue

        hedge_ratio[i] = float(slope)

    # Compute spread: A - beta * B
    spread = np.full(n, np.nan, dtype=float)
    valid_mask = ~np.isnan(hedge_ratio) & ~np.isnan(close_a) & ~np.isnan(close_b)
    spread[valid_mask] = close_a[valid_mask] - hedge_ratio[valid_mask] * close_b[valid_mask]

    # Rolling mean and std for z-score (use population std ddof=0)
    spread_s = pd.Series(spread)
    mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)
    zscore = (spread_s - mean) / std

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore.to_numpy(dtype=float),
    }


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
    Order function for a pairs trading strategy (flexible multi-asset mode).

    Returns a tuple (size, size_type, direction) interpreted by vectorbt's flexible
    order wrapper. Important mapping used here:
        - size_type = 0  -> absolute units
        - direction = 1  -> BUY / LONG
        - direction = 2  -> SELL / SHORT

    Behavior:
        - Entry: when zscore > entry_threshold -> SHORT A, LONG B (1 unit A vs beta units B),
                 when zscore < -entry_threshold -> LONG A, SHORT B
        - Exit: when zscore crosses zero (sign change) -> close both legs
        - Stop-loss: when |zscore| > stop_threshold -> close both legs
        - Position sizing: scale so that Asset A notional == notional_per_leg;
                          Asset B units = hedge_ratio * units_A (maintains beta ratio)

    Notes:
        - This function is called once per asset (col) per bar by the wrapper. It should
          return (np.nan, 0, 0) when no order is to be created for that asset.
        - Uses c.i (bar index), c.col (0 for asset_a, 1 for asset_b), and c.position_now
          which is the current position (in units) for that asset.
    """
    # Defensive checks and extraction
    try:
        i = int(c.i)
    except Exception:
        return (np.nan, 0, 0)

    col = int(getattr(c, "col", 0))

    # Bounds check
    if i < 0 or i >= len(close_a):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan

    z = float(zscore[i]) if (zscore is not None and not np.isnan(zscore[i])) else np.nan
    beta = float(hedge_ratio[i]) if (hedge_ratio is not None and not np.isnan(hedge_ratio[i])) else np.nan

    # Current position in units (may be positive for long, negative for short)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    eps = 1e-9

    # Helper to emit a close order for the current asset
    def _close_current() -> Tuple[float, int, int]:
        if abs(pos_now) < eps:
            return (np.nan, 0, 0)
        size = float(abs(pos_now))
        direction = 2 if pos_now > 0 else 1  # sell to close long; buy to cover short
        return (size, 0, int(direction))

    # STOP-LOSS: close both legs if |zscore| > stop_threshold
    if not np.isnan(z) and abs(z) > stop_threshold:
        return _close_current()

    # EXIT: z-score crosses zero -> close both legs
    prev_z = np.nan
    if i > 0:
        prev_z = float(zscore[i - 1]) if (zscore is not None and not np.isnan(zscore[i - 1])) else np.nan

    crossed_zero = False
    if not np.isnan(prev_z) and not np.isnan(z):
        if (prev_z < 0 and z >= 0) or (prev_z > 0 and z <= 0):
            crossed_zero = True

    if crossed_zero:
        return _close_current()

    # ENTRY: require valid prices and beta
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0 or np.isnan(beta):
        return (np.nan, 0, 0)

    # Units for asset A based on fixed notional per leg
    units_a = float(notional_per_leg / price_a)

    # Are we currently flat on this asset?
    is_flat = abs(pos_now) < eps

    # Long/Short mapping: 1 == BUY, 2 == SELL
    if not np.isnan(z) and z > entry_threshold:
        # SHORT A, LONG B (A -> sell, B -> buy)
        if col == 0:
            # Asset A
            if is_flat:
                if units_a <= 0:
                    return (np.nan, 0, 0)
                return (float(units_a), 0, 2)
            return (np.nan, 0, 0)
        else:
            # Asset B
            if is_flat:
                units_b_raw = units_a * beta
                size = float(abs(units_b_raw))
                if size <= 0:
                    return (np.nan, 0, 0)
                # Desired is LONG on B; if units_b_raw is negative that flips direction
                direction = 1 if units_b_raw >= 0 else 2
                return (size, 0, int(direction))
            return (np.nan, 0, 0)

    if not np.isnan(z) and z < -entry_threshold:
        # LONG A, SHORT B (A -> buy, B -> sell)
        if col == 0:
            # Asset A
            if is_flat:
                if units_a <= 0:
                    return (np.nan, 0, 0)
                return (float(units_a), 0, 1)
            return (np.nan, 0, 0)
        else:
            # Asset B
            if is_flat:
                units_b_raw = units_a * beta
                size = float(abs(units_b_raw))
                if size <= 0:
                    return (np.nan, 0, 0)
                # Desired is SHORT on B; if units_b_raw is negative that flips direction
                direction = 2 if units_b_raw >= 0 else 1
                return (size, 0, int(direction))
            return (np.nan, 0, 0)

    # No order
    return (np.nan, 0, 0)
