"""
Pairs trading strategy: compute_spread_indicators and order_func

Exports:
- compute_spread_indicators(close_a, close_b=None, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg) -> tuple

Notes:
- Uses rolling/expanding OLS for hedge ratio (no lookahead).
- Hedge ratio is computed using only past and current data for each timestamp.
- Z-score uses rolling mean/std with a lookback (uses available data when not enough points).
- order_func is designed for vectorbt flexible multi-asset wrapper used in the backtest runner.

Author: Generated to satisfy backtest harness tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    close_b: Optional[Union[np.ndarray, pd.Series]] = None,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    Args:
        close_a: Price series for asset A. Can be numpy array, pandas Series, or DataFrame (if DataFrame, second asset
                 may be provided in close_b or as the second column / columns named 'asset_a'/'asset_b').
        close_b: Price series for asset B (optional if close_a is a DataFrame containing both assets).
        hedge_lookback: Window length for rolling OLS regression to compute hedge ratio. If not enough points are
                        available, an expanding regression is used (uses all data up to current bar).
        zscore_lookback: Window length for rolling mean/std when computing z-score. Uses available data when fewer
                         points than window are present.

    Returns:
        Dict with keys:
            - 'zscore': numpy array of z-scores (same length as inputs)
            - 'hedge_ratio': numpy array of hedge ratio estimates (same length as inputs)
            - 'spread': numpy array of spreads (same length as inputs)

    The implementation avoids lookahead by computing all statistics using only historical (<= t) data for time t.
    """
    # Accept DataFrame input for convenience
    if close_b is None:
        if isinstance(close_a, pd.DataFrame):
            df = close_a.copy()
            # Try to find asset columns
            if "asset_a" in df.columns and "asset_b" in df.columns:
                pa = df["asset_a"].values.astype(float)
                pb = df["asset_b"].values.astype(float)
            else:
                # Fallback: use first two columns
                if df.shape[1] < 2:
                    raise ValueError("DataFrame input must contain two columns for assets A and B")
                pa = df.iloc[:, 0].values.astype(float)
                pb = df.iloc[:, 1].values.astype(float)
        elif isinstance(close_a, np.ndarray):
            # If a 2D numpy array with shape (n,2)
            if close_a.ndim == 2 and close_a.shape[1] == 2:
                pa = close_a[:, 0].astype(float)
                pb = close_a[:, 1].astype(float)
            else:
                raise ValueError("close_b must be provided when close_a is a 1D numpy array")
        else:
            # If it's a sequence like list of tuples
            arr = np.asarray(close_a)
            if arr.ndim == 2 and arr.shape[1] == 2:
                pa = arr[:, 0].astype(float)
                pb = arr[:, 1].astype(float)
            else:
                raise ValueError("close_b must be provided when close_a does not contain both assets")
    else:
        pa = np.asarray(close_a).astype(float)
        pb = np.asarray(close_b).astype(float)

    if pa.shape[0] != pb.shape[0]:
        raise ValueError("close_a and close_b must have the same length")

    n = pa.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # We will keep the last valid slope to fill degenerate windows
    last_slope = 0.0

    # Compute rolling (expanding when not enough points) OLS slope (hedge ratio) up to and including t
    for t in range(n):
        start = 0 if hedge_lookback <= 0 else max(0, t - hedge_lookback + 1)
        x = pb[start : t + 1]
        y = pa[start : t + 1]
        # Need at least 2 points to compute slope
        if x.size < 2:
            slope = last_slope
        else:
            # Guard against constant x (zero variance) which causes linregress to fail
            if np.allclose(x, x[0]):
                slope = last_slope
            else:
                try:
                    res = linregress(x, y)
                    slope = float(res.slope)
                    if not np.isfinite(slope):
                        slope = last_slope
                except Exception:
                    slope = last_slope
        hedge_ratio[t] = slope
        last_slope = slope
        spread[t] = pa[t] - slope * pb[t]

        # Rolling mean/std for z-score using available history up to t
        w = zscore_lookback if zscore_lookback > 0 else (t + 1)
        start_z = max(0, t - w + 1)
        window = spread[start_z : t + 1]
        # Compute mean/std with population std (ddof=0)
        mean = float(np.nanmean(window)) if window.size > 0 else 0.0
        std = float(np.nanstd(window)) if window.size > 0 else 0.0
        if not np.isfinite(std) or std < 1e-12:
            zscore[t] = 0.0
        else:
            zscore[t] = (spread[t] - mean) / std

    return {
        "zscore": zscore,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
    }


def order_func(
    c: Any,
    close_a: Union[np.ndarray, pd.Series],
    close_b: Union[np.ndarray, pd.Series],
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    notional_per_leg: float = 10_000.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset mode (vectorbt). Returns a tuple (size, size_type, direction).

    - size_type: 0 => SIZE (number of shares)
    - direction:  1 => buy, 2 => sell

    The function computes target positions (in number of shares) for both assets based on z-score and hedge ratio,
    then returns the required trade (delta) for the asset indicated by c.col.

    Args:
        c: order context with attributes .i (bar index), .col (column index), and .position_now (current position for that column).
        close_a, close_b: price arrays for assets A and B (numpy arrays or Series). Must align with zscore/hedge_ratio.
        zscore, hedge_ratio: indicator arrays computed by compute_spread_indicators.
        entry_threshold: threshold to open positions (default 2.0)
        exit_threshold: unused threshold (exit logic uses crossing 0.0 per spec)
        stop_threshold: stop-loss threshold (default 3.0)
        notional_per_leg: fixed notional per leg (dollars). Used to derive share sizes for asset A; asset B size is scaled by hedge ratio.

    Returns:
        (size, size_type, direction) where size_type=0 indicates size is number of shares and direction=1 buys shares, 2 sells.
        If no order is required, returns (np.nan, 0, 0) which the wrapper interprets as NoOrder for that column.
    """
    # Extract simple context values
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive conversion
    pa = np.asarray(close_a).astype(float)
    pb = np.asarray(close_b).astype(float)

    n = len(pa)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if (zscore is not None and len(zscore) > i) else 0.0
    hr = float(hedge_ratio[i]) if (hedge_ratio is not None and len(hedge_ratio) > i) else np.nan

    # Current position for this column (number of shares). If unavailable, assume 0.
    pos_now = getattr(c, "position_now", 0.0)
    try:
        pos_now = 0.0 if pos_now is None else float(pos_now)
    except Exception:
        pos_now = 0.0

    # Determine previous z-score for exit-on-cross detection
    prev_z = float(zscore[i - 1]) if (i > 0 and len(zscore) > i - 1) else None

    # If hedge ratio is not finite, do nothing
    if not np.isfinite(hr) or hr == 0.0:
        return (np.nan, 0, 0)

    price_a = float(pa[i])
    price_b = float(pb[i])

    # Avoid division by zero
    if not np.isfinite(price_a) or price_a <= 0:
        return (np.nan, 0, 0)
    if not np.isfinite(price_b) or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute base share size for asset A using fixed notional per leg
    base_a_shares = notional_per_leg / price_a

    # Target positions (in shares) for this bar
    target_a: Optional[float] = None
    target_b: Optional[float] = None

    # Stop-loss: if |z| > stop_threshold -> close both
    if abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry conditions
        if z > entry_threshold:
            # Short A, Long B
            target_a = -base_a_shares
            target_b = +abs(hr) * base_a_shares
        elif z < -entry_threshold:
            # Long A, Short B
            target_a = +base_a_shares
            target_b = -abs(hr) * base_a_shares
        else:
            # Exit when z-score crosses zero
            if prev_z is not None and (prev_z * z) < 0:
                target_a = 0.0
                target_b = 0.0
            else:
                # No target change
                target_a = None
                target_b = None

    # Select appropriate target for this column
    target = target_a if col == 0 else target_b

    # If no action for this column
    if target is None:
        return (np.nan, 0, 0)

    # Compute delta (how many shares to trade)
    delta = float(target - pos_now)

    # If delta is effectively zero, no order
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    size_to_trade = abs(delta)
    size_type = 0  # SIZE (number of shares)
    # Direction mapping: vectorbt expects 1 for buy/long and 2 for sell/short
    direction = 1 if delta > 0 else 2

    # Return tuple (size, size_type, direction)
    return (size_to_trade, int(size_type), int(direction))
