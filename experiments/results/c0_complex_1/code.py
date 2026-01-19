"""
Pairs trading strategy utilities for vectorbt backtesting.

Exports:
- compute_spread_indicators
- order_func

Notes:
- Hedge ratio is computed by rolling OLS (lookback periods).
- Spread = Price_A - hedge_ratio * Price_B
- Z-score computed with rolling mean/std (zscore_lookback)
- Entry/Exit/Stop-loss logic implemented in order_func

CRITICAL: This module does NOT use numba anywhere.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


# Track how many orders we've returned for a given bar index to avoid infinite loops
_orders_issued_per_bar: Dict[int, int] = {}


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs trading strategy.

    Args:
        close_a: Prices for asset A (1D array-like)
        close_b: Prices for asset B (1D array-like)
        hedge_lookback: Lookback window (in bars) for rolling OLS to estimate hedge ratio
        zscore_lookback: Lookback window for rolling mean/std to compute z-score of the spread

    Returns:
        dict with keys:
            - "hedge_ratio": np.ndarray of same length as inputs with rolling OLS slope
            - "zscore": np.ndarray with rolling z-score of the spread
            - "spread": np.ndarray with the computed spread (Price_A - hedge_ratio * Price_B)
            - "roll_mean": rolling mean of spread
            - "roll_std": rolling std of spread

    Notes:
        - Missing values are preserved as np.nan
        - The function requires at least `hedge_lookback` observations to produce a hedge_ratio
          and at least `zscore_lookback` observations to produce a z-score.
    """

    # Convert to numpy arrays of floats
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    # Prepare output arrays
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS: regress A on B in each rolling window
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")

    for end_idx in range(hedge_lookback - 1, n):
        start_idx = end_idx - hedge_lookback + 1
        y = a[start_idx : end_idx + 1]
        x = b[start_idx : end_idx + 1]

        # Require at least two non-NaN points to run regression
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            hedge_ratio[end_idx] = np.nan
            continue

        try:
            # linregress may fail if x is constant; handle with try/except
            res = linregress(x[mask], y[mask])
            hedge_ratio[end_idx] = float(res.slope)
        except Exception:
            hedge_ratio[end_idx] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = a[valid_hr] - hedge_ratio[valid_hr] * b[valid_hr]

    # Rolling mean/std for z-score using pandas for convenience
    s = pd.Series(spread)
    roll_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) to match typical z-score definition
    roll_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(roll_mean)) & (~np.isnan(roll_std)) & (roll_std > 0)
    zscore[valid_z] = (spread[valid_z] - roll_mean[valid_z]) / roll_std[valid_z]

    # Return dictionary (ensure arrays are numpy arrays)
    return {
        "hedge_ratio": np.asarray(hedge_ratio, dtype=float),
        "zscore": np.asarray(zscore, dtype=float),
        "spread": np.asarray(spread, dtype=float),
        "roll_mean": np.asarray(roll_mean, dtype=float),
        "roll_std": np.asarray(roll_std, dtype=float),
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
) -> Tuple[float, int, int]:
    """
    Order function for vectorbt flexible multi-asset backtest.

    The function follows these rules:
    - Entry: if zscore > entry_threshold -> Short A, Long B (hedge_ratio units)
             if zscore < -entry_threshold -> Long A, Short B (hedge_ratio units)
    - Exit: if zscore crosses 0 -> close positions
    - Stop-loss: if |zscore| > stop_threshold -> close positions
    - Position sizing: fixed notional = $10,000 per leg. We compute the unit size for asset A
      as notional / price_A and size for asset B as abs(hedge_ratio) * units_A. Direction for B is
      determined by the entry signal (long or short B).

    Args:
        c: Order context provided by vectorbt (or the wrapper). Expected attributes used:
           - i: current bar index
           - col: current column index (0 for asset_a, 1 for asset_b)
           - position_now: current position size for this asset (may be 0)
           - cash_now or value_now: current cash (not strictly required here)
        close_a, close_b: price arrays
        zscore: zscore array computed by indicators
        hedge_ratio: hedge ratio array computed by indicators
        entry_threshold, exit_threshold, stop_threshold: thresholds

    Returns:
        (size, size_type, direction)
        - size: float number of units to buy/sell (absolute). Return np.nan to indicate no order.
        - size_type: int, 0 indicates absolute units
        - direction: int, 1 for BUY/long, 2 for SELL/short

    Important: This function does NOT use numba and returns plain Python objects.
    """

    # Configuration
    NOTIONAL_PER_LEG = 10_000.0
    SIZE_TYPE_ABSOLUTE = 0
    DIR_BUY = 1
    DIR_SELL = 2

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Limit orders per bar to avoid runaway order generation
    count = _orders_issued_per_bar.get(i, 0)
    if count >= 2:
        return (float("nan"), 0, 0)

    # Safely get prices and indicators at current index
    try:
        price_a = float(close_a[i])
    except Exception:
        price_a = float("nan")
    try:
        price_b = float(close_b[i])
    except Exception:
        price_b = float("nan")

    # Current zscore and hedge_ratio
    z = float(zscore[i]) if (i >= 0 and i < len(zscore)) else float("nan")
    hr = float(hedge_ratio[i]) if (i >= 0 and i < len(hedge_ratio)) else float("nan")

    # Current position for this asset (units). Default to 0
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Helper: return no order
    def no_order():
        return (float("nan"), 0, 0)

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return no_order()

    # Compute previous z to detect zero-crossing (exit), if available
    z_prev = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else float("nan")

    # Stop-loss: if |z| > stop_threshold -> close any existing position
    if abs(z) > stop_threshold:
        if pos_now == 0:
            return no_order()
        # Close current position: if long -> SELL, if short -> BUY
        if pos_now > 0:
            _orders_issued_per_bar[i] = count + 1
            return (abs(pos_now), SIZE_TYPE_ABSOLUTE, DIR_SELL)
        else:
            _orders_issued_per_bar[i] = count + 1
            return (abs(pos_now), SIZE_TYPE_ABSOLUTE, DIR_BUY)

    # Exit: z-score crosses 0.0 (from positive to <=0 or from negative to >=0)
    crossed_to_zero = False
    if not np.isnan(z_prev):
        if (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold):
            crossed_to_zero = True

    if crossed_to_zero:
        if pos_now == 0:
            return no_order()
        _orders_issued_per_bar[i] = count + 1
        if pos_now > 0:
            return (abs(pos_now), SIZE_TYPE_ABSOLUTE, DIR_SELL)
        else:
            return (abs(pos_now), SIZE_TYPE_ABSOLUTE, DIR_BUY)

    # Entry signals
    # Compute unit sizes based on NOTIONAL_PER_LEG and current prices
    # If price is invalid or zero, skip
    if price_a <= 0 or price_b <= 0:
        return no_order()

    units_a = NOTIONAL_PER_LEG / price_a

    # If hedge ratio is extremely small, avoid opening a degenerate leg
    if abs(hr) < 1e-12:
        # If hedge ratio is effectively zero, we cannot compute B units reliably -> skip
        return no_order()

    units_b = abs(hr) * units_a

    # Entry: z > entry_threshold -> Short A, Long B
    if z > entry_threshold:
        # Asset A (col 0) -> short
        if col == 0:
            # Only enter if no existing position
            if pos_now == 0:
                _orders_issued_per_bar[i] = count + 1
                return (units_a, SIZE_TYPE_ABSOLUTE, DIR_SELL)
            return no_order()
        # Asset B (col 1) -> long
        else:
            if pos_now == 0:
                _orders_issued_per_bar[i] = count + 1
                return (units_b, SIZE_TYPE_ABSOLUTE, DIR_BUY)
            return no_order()

    # Entry: z < -entry_threshold -> Long A, Short B
    if z < -entry_threshold:
        if col == 0:
            if pos_now == 0:
                _orders_issued_per_bar[i] = count + 1
                return (units_a, SIZE_TYPE_ABSOLUTE, DIR_BUY)
            return no_order()
        else:
            if pos_now == 0:
                _orders_issued_per_bar[i] = count + 1
                return (units_b, SIZE_TYPE_ABSOLUTE, DIR_SELL)
            return no_order()

    # Otherwise, no order
    return no_order()
