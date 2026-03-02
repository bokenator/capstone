"""
Pairs trading strategy utilities for vectorbt backtester.

Exports:
- compute_spread_indicators
- order_func

Notes:
- No numba is used.
- The order function returns plain Python tuples (size, size_type, direction).

Author: AI assistant
"""
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio, spread and z-score for a pair of assets.

    Hedge ratio is computed as rolling OLS slope (regressing A on B) using
    beta = cov(A, B) / var(B) over a lookback window.

    Z-score is computed from the spread using rolling mean and std.

    Args:
        close_a: Prices for asset A (1D numpy array).
        close_b: Prices for asset B (1D numpy array).
        hedge_lookback: Lookback window for hedge ratio regression (default 60).
        zscore_lookback: Lookback window for z-score (default 20).

    Returns:
        Dict containing numpy arrays (same length as inputs) for keys:
        - "hedge_ratio"
        - "spread"
        - "rolling_mean"
        - "rolling_std"
        - "zscore"

    Handles NaNs and warmup periods by returning NaN for indices where the
    rolling windows are not fully populated.
    """
    # Validate inputs
    if close_a is None or close_b is None:
        raise ValueError("close_a and close_b must be provided")

    a = pd.Series(np.asarray(close_a).astype(float)).reset_index(drop=True)
    b = pd.Series(np.asarray(close_b).astype(float)).reset_index(drop=True)

    if len(a) != len(b):
        raise ValueError("close_a and close_b must have the same length")

    # Rolling covariance and variance (use min_periods = hedge_lookback to match exact lookback)
    # Use sample covariance/variance (ddof=1) which is pandas default for .var()
    cov_ab = a.rolling(window=hedge_lookback, min_periods=hedge_lookback).cov(b)
    var_b = b.rolling(window=hedge_lookback, min_periods=hedge_lookback).var()

    # Avoid division by zero
    hedge_ratio = cov_ab / var_b
    hedge_ratio = hedge_ratio.replace([np.inf, -np.inf], np.nan)

    # Spread
    spread = a - hedge_ratio * b

    # Rolling mean/std for spread
    rolling_mean = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    zscore = (spread - rolling_mean) / rolling_std

    # Convert back to numpy arrays
    out = {
        "hedge_ratio": hedge_ratio.to_numpy(dtype=float),
        "spread": spread.to_numpy(dtype=float),
        "rolling_mean": rolling_mean.to_numpy(dtype=float),
        "rolling_std": rolling_std.to_numpy(dtype=float),
        "zscore": zscore.to_numpy(dtype=float),
    }

    return out


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
    """Order function for flexible multi-asset vectorbt backtest (no numba).

    This function is called for each asset (column) separately. It returns a
    tuple (size, size_type, direction) where:
      - size: float (amount or value depending on size_type) or np.nan for NoOrder
      - size_type: int matching vectorbt.portfolio.enums.SizeType
          (0=Amount, 1=Value, ...)
      - direction: int matching vectorbt.portfolio.enums.TradeDirection
          (0=Long/Buy, 1=Short/Sell)

    Strategy logic implemented:
      - Entry when zscore > entry_threshold: Short A, Long B
      - Entry when zscore < -entry_threshold: Long A, Short B
      - Exit when zscore crosses exit_threshold (typically 0.0) OR |zscore| > stop_threshold
      - Position sizing: fixed notional_per_leg dollars on asset A; asset B sized
        in units to match hedge ratio (units_B = |hedge_ratio| * units_A).

    Notes:
      - If hedge_ratio is NaN or required prices are NaN, no order is returned.
      - Uses SizeType.Value (1) for dollar-based orders and SizeType.Amount (0)
        for unit-based orders when closing or sizing B leg.

    Args:
        c: Order context (has attributes i, col, position_now). Provided by wrapper.
        close_a, close_b: 1D numpy arrays of close prices.
        zscore: 1D numpy array of z-score values.
        hedge_ratio: 1D numpy array of hedge ratios.
        entry_threshold: Threshold to enter trades (default 2.0).
        exit_threshold: Threshold to exit trades (default 0.0).
        stop_threshold: Stop-loss threshold (default 3.0).
        notional_per_leg: Fixed dollar notional to allocate to asset A (default 10k).

    Returns:
        (size, size_type, direction) tuple. Use (np.nan, 0, 0) to indicate no order.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safe access helpers
    def safe_get(arr: np.ndarray, idx: int) -> float:
        try:
            return float(arr[idx])
        except Exception:
            return float("nan")

    price_a = safe_get(close_a, i)
    price_b = safe_get(close_b, i)
    z = safe_get(zscore, i)
    hr = safe_get(hedge_ratio, i)

    # If any critical value is nan, do nothing
    if np.isnan(z) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Treat tiny positions as zero
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos = float(pos_now)
    except Exception:
        # If position_now is an array-like, try first element
        try:
            pos = float(pos_now[0])
        except Exception:
            pos = 0.0

    if abs(pos) < 1e-12:
        pos = 0.0

    # Previous z for crossover detection
    prev_z = safe_get(zscore, i - 1) if i > 0 else float("nan")

    # Entry/exit/stop signals
    is_entry_shortA_longB = z > entry_threshold
    is_entry_longA_shortB = z < -entry_threshold
    is_stop_loss = abs(z) > stop_threshold

    crossed_zero = False
    if not (np.isnan(prev_z) or np.isnan(z)):
        # Cross when previous was on opposite side or touches the exit_threshold
        if (prev_z < 0 and z >= exit_threshold) or (prev_z > 0 and z <= exit_threshold):
            crossed_zero = True

    # Helper enum values (use ints so they are serializable by the wrapper)
    SIZE_AMOUNT = 0  # SizeType.Amount
    SIZE_VALUE = 1   # SizeType.Value
    DIR_LONG = 0     # TradeDirection.Long / Buy
    DIR_SHORT = 1    # TradeDirection.Short / Sell

    # If no position, consider opening
    if pos == 0.0:
        # ENTRY: Short A, Long B when zscore > entry
        if is_entry_shortA_longB:
            if col == 0:
                # Short Asset A with fixed notional
                return (float(notional_per_leg), SIZE_VALUE, DIR_SHORT)
            else:
                # Long Asset B sized by hedge ratio (units)
                if np.isnan(hr) or hr == 0 or np.isnan(price_a) or price_a == 0:
                    return (np.nan, 0, 0)
                units_a = notional_per_leg / price_a
                units_b = abs(hr) * units_a
                if not np.isfinite(units_b) or units_b <= 0:
                    return (np.nan, 0, 0)
                return (float(units_b), SIZE_AMOUNT, DIR_LONG)

        # ENTRY: Long A, Short B when zscore < -entry
        if is_entry_longA_shortB:
            if col == 0:
                # Long Asset A with fixed notional
                return (float(notional_per_leg), SIZE_VALUE, DIR_LONG)
            else:
                # Short Asset B sized by hedge ratio (units)
                if np.isnan(hr) or hr == 0 or np.isnan(price_a) or price_a == 0:
                    return (np.nan, 0, 0)
                units_a = notional_per_leg / price_a
                units_b = abs(hr) * units_a
                if not np.isfinite(units_b) or units_b <= 0:
                    return (np.nan, 0, 0)
                return (float(units_b), SIZE_AMOUNT, DIR_SHORT)

        # No entry
        return (np.nan, 0, 0)

    # If we have a position, consider closing on exit or stop-loss
    if pos != 0.0:
        if crossed_zero or is_stop_loss:
            # Close entire position by specifying amount equal to abs(current units)
            size_to_close = abs(pos)
            if size_to_close <= 0 or not np.isfinite(size_to_close):
                return (np.nan, 0, 0)
            # If currently long (pos>0) -> Sell to close; if short (pos<0) -> Buy to close
            direction = DIR_SHORT if pos > 0 else DIR_LONG
            return (float(size_to_close), SIZE_AMOUNT, int(direction))

        # Otherwise do nothing (no pyramiding / scaling)
        return (np.nan, 0, 0)
