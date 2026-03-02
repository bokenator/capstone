import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from scipy import stats


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio using OLS regression and z-score of the spread.

    Parameters
    ----------
    close_a : np.ndarray
        Close prices for asset A (dependent variable in regression).
    close_b : np.ndarray
        Close prices for asset B (independent variable in regression).
    hedge_lookback : int
        Lookback window (in bars) for rolling OLS regression to estimate hedge ratio.
    zscore_lookback : int
        Lookback window (in bars) for rolling mean/std when computing z-score of the spread.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing at least:
            - 'hedge_ratio': np.ndarray of same length as inputs with rolling hedge ratios (NaN for warmup)
            - 'spread': np.ndarray of spread values (NaN where hedge_ratio is NaN)
            - 'zscore': np.ndarray of z-score values (NaN for warmup)
            - 'rolling_mean': np.ndarray of rolling mean of spread
            - 'rolling_std': np.ndarray of rolling std of spread

    Notes
    -----
    - Rolling OLS regresses price_A ~ beta * price_B + intercept and returns beta as hedge ratio.
    - Z-score is computed as (spread - rolling_mean) / rolling_std with population std (ddof=0).
    - Handles NaNs and warmup periods by returning NaNs for those indices.
    """

    ca = np.asarray(close_a, dtype=float).flatten()
    cb = np.asarray(close_b, dtype=float).flatten()

    if ca.shape != cb.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = ca.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 1:
        raise ValueError("hedge_lookback must be >= 1")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be >= 1")

    # Rolling OLS: regress A on B -> slope is hedge ratio
    if hedge_lookback == 1:
        # Degenerate case: slope = A / B (where defined)
        with np.errstate(divide='ignore', invalid='ignore'):
            hr = ca / cb
            hr[~np.isfinite(hr)] = np.nan
            hedge_ratio[:] = hr
    else:
        for i in range(hedge_lookback - 1, n):
            x = cb[i - hedge_lookback + 1 : i + 1]
            y = ca[i - hedge_lookback + 1 : i + 1]
            # Require full window without NaNs for the regression (conservative)
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            # Design matrix with intercept
            A = np.vstack([x, np.ones_like(x)]).T
            try:
                # Solve least squares for slope and intercept
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                slope = float(sol[0])
                hedge_ratio[i] = slope
            except Exception:
                # Fallback to scipy linregress
                try:
                    slope, intercept, r, p, se = stats.linregress(x, y)
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan

    # Compute spread where hedge_ratio is available
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = ca[valid_hr] - hedge_ratio[valid_hr] * cb[valid_hr]

    # Rolling mean and std for z-score using pandas rolling (requires full window)
    s = pd.Series(spread)
    rolling_mean = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    # Use population std (ddof=0) for z-score stability
    rolling_std = s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).to_numpy()

    zscore = np.full(n, np.nan, dtype=float)
    valid_z = (~np.isnan(spread)) & (~np.isnan(rolling_mean)) & (~np.isnan(rolling_std)) & (rolling_std > 0)
    zscore[valid_z] = (spread[valid_z] - rolling_mean[valid_z]) / rolling_std[valid_z]

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
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
    """
    Order function for flexible multi-asset vectorbt backtest (pairs trading).

    This function is called once per asset (column) per bar. It returns a tuple
    (size, size_type, direction) describing the order for that asset on that bar.

    Assumptions / Conventions
    - c.i: current integer bar index
    - c.col: column index (0 = asset A, 1 = asset B)
    - c.position_now: current position in asset units (can be 0.0)

    Sizing and Signals
    - Uses a simple target-based approach: compute desired target units for each
      asset given the current z-score and hedge ratio, then issue an order to
      move from current position to the target (difference).
    - Fixed notional sizing per leg: base_size_a = notional_per_leg / price_A.
      The target for asset B follows the hedge ratio: target_B = -target_A * hedge_ratio
      (so that one "spread unit" is 1 A and hedge_ratio * B).

    Entry / Exit / Stop-loss
    - If zscore > entry_threshold: short A, long B
    - If zscore < -entry_threshold: long A, short B
    - If |zscore| > stop_threshold: close both positions (stop-loss)
    - Otherwise (including zscore crossing 0): target is zero (close)

    Returns
    -------
    (size, size_type, direction)
        - size: number of asset units to trade (absolute value). Return NaN for no order.
        - size_type: integer code for how size is interpreted by vectorbt. We use 0
          (assumed to mean absolute number of shares/units).
        - direction: integer code for buy/sell. We use 1 for buy (increase position),
          2 for sell (decrease position / go short).

    Notes
    -----
    - The wrapper provided by the backtest runner will convert this tuple into
      a numba order object. We therefore return simple Python types.
    - The mapping (size_type=0, direction=1/2) matches common flexible-order
      examples where 1 indicates buy and 2 indicates sell.
    """

    # Defensive typing
    i = int(getattr(c, "i"))
    col = int(getattr(c, "col", 0))

    # Validate index bounds
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Extract necessary values for this bar
    price_a = float(close_a[i]) if not np.isnan(close_a[i]) else np.nan
    price_b = float(close_b[i]) if not np.isnan(close_b[i]) else np.nan
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # Current position in units (per-column)
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = float(pos_now)
    except Exception:
        pos_now = 0.0

    # Warmup / invalid data -> no order
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Basic sanity checks on prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute base sizing for asset A (number of units corresponding to the notional)
    base_size_a = float(notional_per_leg) / price_a

    # Default targets
    if z > entry_threshold:
        # Short A, Long B
        target_a = -base_size_a
    elif z < -entry_threshold:
        # Long A, Short B
        target_a = base_size_a
    else:
        # Between entry thresholds -> target flat (exit)
        target_a = 0.0

    # Apply stop-loss: if |z| > stop_threshold, close positions immediately
    if abs(z) > stop_threshold:
        target_a = 0.0

    # Target for B derived from A and hedge ratio (one spread unit = 1*A + hr*B)
    target_b = -target_a * hr

    target = target_a if col == 0 else target_b

    # Difference between target and current position -> how much to trade
    diff = target - pos_now

    # If nothing to do (within tolerance) -> no order
    if abs(diff) < 1e-8:
        return (np.nan, 0, 0)

    size = abs(diff)
    # Map diff sign: positive -> buy, negative -> sell
    direction = 1 if diff > 0 else 2

    # size_type = 0 -> absolute number of units (assumption)
    return (float(size), 0, int(direction))
