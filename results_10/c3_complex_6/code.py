import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs trading strategy.

    Args:
        close_a: Prices of asset A (1D array)
        close_b: Prices of asset B (1D array)
        hedge_lookback: Lookback window for rolling OLS to estimate hedge ratio. If there
            are fewer points than this at the beginning, an expanding-window estimate is used.
        zscore_lookback: Lookback window for rolling mean/std of the spread used to compute z-score.

    Returns:
        Dictionary with keys:
            - "hedge_ratio": np.ndarray of same length as inputs with per-bar hedge ratio (slope)
            - "zscore": np.ndarray of same length with z-score of the spread
            - "spread": np.ndarray of the spread (A - beta * B)

    Notes:
        - Implemented so that no future data is used to compute indicators at time t.
        - Uses min_periods=1 for rolling mean/std so early bars are defined (avoids NaNs).
    """
    # Convert inputs to numpy arrays (1d) and validate
    close_a = np.asarray(close_a, dtype=float).ravel()
    close_b = np.asarray(close_b, dtype=float).ravel()

    if close_a.shape != close_b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = len(close_a)
    if n == 0:
        return {"hedge_ratio": np.array([]), "zscore": np.array([]), "spread": np.array([])}

    # Prepare arrays
    hedge_ratio = np.zeros(n, dtype=float)

    # Rolling OLS (slope only) using expanding windows until lookback is reached
    for t in range(n):
        start = max(0, t - hedge_lookback + 1)
        x = close_b[start : t + 1]
        y = close_a[start : t + 1]

        # If variance of x is 0 (constant series) or length 1, set slope to 0.0
        if x.size < 2:
            hedge_ratio[t] = 0.0
            continue

        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom <= 0.0:
            hedge_ratio[t] = 0.0
        else:
            hedge_ratio[t] = np.sum((x - x_mean) * (y - y_mean)) / denom

    # Compute spread using the contemporaneous hedge ratio
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score with min_periods=1 to avoid NaNs early on
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use population std (ddof=0) so std is defined when window size is 1
    roll_std = spread_series.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Avoid division by zero
    eps = 1e-8
    zscore = (spread - roll_mean) / (roll_std + eps)

    return {"hedge_ratio": hedge_ratio, "zscore": zscore, "spread": spread}



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
    Order function for a pairs trading strategy in flexible multi-asset mode.

    This function is called once per asset (column) per bar. It is stateless and uses only
    the provided arrays up to the current index c.i to make decisions (no lookahead).

    Returns a tuple (size, size_type, direction):
        - size: absolute size (number of units) for the order (or np.nan to indicate NoOrder)
        - size_type: integer encoding how to interpret size. We use 1 to indicate an absolute
                     order size (change in units), i.e. place an order for `size` units.
        - direction: 0 for flexible (engine chooses proper side based on position change)

    Positioning logic:
        - Entry: when zscore > entry_threshold => short A, long B (scaled by hedge_ratio)
                 when zscore < -entry_threshold => long A, short B (scaled by hedge_ratio)
        - Exit: when zscore crosses zero (sign flip) OR abs(zscore) > stop_threshold => close both legs
        - Position sizing: base units = notional_per_leg / price_A
                          pos_A = -sign(zscore) * base_units
                          pos_B =  sign(zscore) * hedge_ratio * base_units
                          (this enforces pos_B ≈ -hedge_ratio * pos_A)

    Notes:
        - The function returns np.nan for size when no order is needed for the current asset.
        - Uses only values at index c.i (and c.i-1 for z-score crossing detection).
    """
    # Defensive access to context attributes
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0)) if getattr(c, "col", None) is not None else 0
    position_now = float(getattr(c, "position_now", 0.0))

    # Safety bounds
    n = len(zscore)
    if i < 0 or i >= n:
        # Invalid bar index - return NoOrder
        return (float("nan"), 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Read indicators at or before i (no future use)
    z_t = float(zscore[i]) if not np.isnan(zscore[i]) else 0.0
    hedge_t = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else 0.0

    # Detect crossings of zero (use previous zscore if available)
    crossed_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        z_prev = float(zscore[i - 1])
        # A crossing happens when signs are opposite
        if z_prev * z_t < 0:
            crossed_zero = True

    # Stop-loss if |z| exceeds threshold
    stop_loss = abs(z_t) > stop_threshold

    # Determine targets
    # Base units derived from asset A price to keep a reference sizing
    base_units = 0.0
    if price_a > 0 and np.isfinite(price_a):
        base_units = float(notional_per_leg) / price_a

    target_a = None
    target_b = None

    # If stop loss or crossing zero -> close both legs
    if stop_loss or crossed_zero:
        target_a = 0.0
        target_b = 0.0
    # Entry conditions
    elif z_t > entry_threshold:
        # Short A, Long B (scaled by hedge_ratio)
        target_a = -base_units
        target_b = hedge_t * base_units
    elif z_t < -entry_threshold:
        # Long A, Short B
        target_a = base_units
        target_b = -hedge_t * base_units
    else:
        # No new signal; keep existing positions
        # We do not issue orders unless a change is needed.
        target_a = None
        target_b = None

    # Helper to build order tuple for this column
    def build_order_for_col(col_idx: int) -> Tuple[float, int, int]:
        nonlocal target_a, target_b, position_now
        if col_idx == 0:
            tgt = target_a
            price = price_a
        else:
            tgt = target_b
            price = price_b

        # If target is None, no action for this bar for this asset
        if tgt is None:
            return (float("nan"), 0, 0)

        # Compute delta = desired_change = target - current_position
        delta = tgt - position_now

        # If change is negligible -> no order
        tol = 1e-8
        if abs(delta) <= tol:
            return (float("nan"), 0, 0)

        # size_type=1 -> place an absolute order of this many units (change)
        size_type = 1

        # Use neutral direction (0) and let engine infer side from delta/position
        direction = 0
        size = float(abs(delta))

        return (size, size_type, direction)

    return build_order_for_col(col)
