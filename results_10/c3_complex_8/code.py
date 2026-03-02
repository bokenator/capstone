"""
Pairs trading strategy implementation for vectorbt backtesting.

Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict[str, np.ndarray]
- order_func(ctx, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg)

Notes:
- Rolling OLS hedge ratio computed using past data only (no lookahead).
- Rolling mean/std for z-score computed with past data only.
- Uses simple OLS slope formula (cov/var) with small-denominator protection.
- Order function returns (size, size_type, direction) tuples compatible with vectorbt's from_order_func (flexible mode).

CRITICAL: Does not use numba or vbt.portfolio.nb.* directly.
"""
from typing import Any, Dict, Tuple

import numpy as np


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score for a pair of assets.

    Hedge ratio is computed as the rolling OLS slope (A ~ alpha + beta * B) using
    only past and present data in each window. If there are fewer observations
    than the requested lookback, the regression is computed on the available
    history (min_periods = 2 effectively).

    Z-score is (spread - rolling_mean) / rolling_std where spread = A - beta * B.
    Rolling mean/std use a backward-looking window of zscore_lookback (with
    available-data fallback).

    Returns:
        dict with keys:
        - "zscore": np.ndarray, same length as inputs
        - "hedge_ratio": np.ndarray, same length as inputs
    """
    # Convert to numpy arrays and validate
    a = np.asarray(close_a, dtype=float).ravel()
    b = np.asarray(close_b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Small epsilon to avoid division by zero
    EPS = 1e-8

    # Compute rolling hedge ratio (slope) using only past & current observations
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = b[start : i + 1]
        y = a[start : i + 1]

        if x.size >= 2:
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom < EPS:
                beta = 0.0
            else:
                beta = np.sum((x - x_mean) * (y - y_mean)) / denom
        else:
            # Not enough data to compute regression slope robustly
            beta = 0.0

        hedge_ratio[i] = beta
        spread[i] = a[i] - beta * b[i]

    # Compute rolling mean/std of spread for z-score using backward-looking window
    for i in range(n):
        start = max(0, i - zscore_lookback + 1)
        w = spread[start : i + 1]
        if w.size == 0:
            mu = 0.0
            sigma = 0.0
        else:
            mu = w.mean()
            sigma = w.std(ddof=0)

        if sigma < EPS:
            z = 0.0
        else:
            z = (spread[i] - mu) / sigma

        zscore[i] = z

    return {"zscore": zscore, "hedge_ratio": hedge_ratio}


def order_func(
    ctx: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float,
) -> Tuple[float, int, int]:
    """Order function for flexible vectorbt.from_order_func mode.

    This function is called once per asset per bar. The simulation wrapper will
    call it for both assets on the same bar, so decisions are made based on the
    same z-score and hedge ratio at index `ctx.i`.

    Returns a tuple (size, size_type, direction):
    - size: absolute quantity (number of units) to trade for this asset, or np.nan
            to indicate no order.
    - size_type: integer indicating size interpretation. We use 0 to indicate
                 a raw number of units.
    - direction: 1 for buy (increase position), 2 for sell (decrease position).

    Notes:
    - Entry: z > entry_threshold => Short A, Long B
             z < -entry_threshold => Long A, Short B
    - Exit: z crosses 0 => close both legs
    - Stop-loss: |z| > stop_threshold => close both legs
    - Position sizing: base units = notional_per_leg / price_A; Asset B is scaled
      by the hedge ratio so that position_b ~= -hedge_ratio * position_a.
    """
    i = int(ctx.i)
    col = int(getattr(ctx, "col", getattr(ctx, "column", 0)))

    # Coerce arrays to numpy for safe indexing
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()
    zscore = np.asarray(zscore, dtype=float).ravel()
    hedge_ratio = np.asarray(hedge_ratio, dtype=float).ravel()

    # Defensive bounds
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # current position for this asset (in units)
    pos_now = float(getattr(ctx, "position_now", 0.0))

    z = float(zscore[i]) if not np.isnan(zscore[i]) else 0.0
    beta = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0

    # Detect cross of zero (use sign change from previous bar)
    cross_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        if prev_z * z < 0:
            cross_zero = True

    # Stop-loss or crossover => set both targets to zero
    if (not np.isnan(z) and abs(z) > stop_threshold) or cross_zero:
        target_a = 0.0
        target_b = 0.0
    else:
        # Entry checks
        if z > entry_threshold:
            # Short A, Long B
            base_units = notional_per_leg / price_a if price_a > 0 else 0.0
            target_a = -base_units
            # Ensure B is sized according to hedge ratio: target_b ~= -beta * target_a
            target_b = -beta * target_a
        elif z < -entry_threshold:
            # Long A, Short B
            base_units = notional_per_leg / price_a if price_a > 0 else 0.0
            target_a = base_units
            target_b = -beta * target_a
        else:
            # No signal: do not change positions
            return (np.nan, 0, 0)

    # Select target for this asset
    target = target_a if col == 0 else target_b

    # Compute delta to current position
    delta = target - pos_now

    # If no change, return NoOrder
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    size = float(abs(delta))
    size_type = 0  # interpret size as raw units
    # Direction: 1 -> buy/increase position, 2 -> sell/decrease position
    direction = 1 if delta > 0 else 2

    return (size, int(size_type), int(direction))
