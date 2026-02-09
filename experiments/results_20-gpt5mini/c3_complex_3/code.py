# Pairs trading strategy implementation for vectorbt
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    Args:
        close_a (np.ndarray): Close prices for asset A (1D array).
        close_b (np.ndarray): Close prices for asset B (1D array).
        hedge_lookback (int): Lookback window for rolling OLS hedge ratio. If
            there are fewer than hedge_lookback observations at time t, an
            expanding window (all available past data up to t) is used instead
            to avoid NaNs and lookahead bias.
        zscore_lookback (int): Lookback window for rolling mean/std of the spread.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing at least:
            - 'zscore': z-score of the spread (np.ndarray)
            - 'hedge_ratio': rolling hedge ratio (np.ndarray)
            Additional keys 'spread', 'spread_mean', 'spread_std' are also provided.

    Notes:
        - All calculations are performed using only past and present data at
          each timestamp (no lookahead).
        - If regression is degenerate (zero variance in X), the previous valid
          hedge ratio is carried forward.
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    spread_mean = np.full(n, np.nan, dtype=float)
    spread_std = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Start with a reasonable default hedge ratio to avoid NaNs at the very
    # beginning. We'll overwrite as soon as we can compute a proper regression.
    last_valid_beta = 1.0

    for i in range(n):
        # Determine regression window (use past data up to i inclusive)
        start_reg = max(0, i + 1 - hedge_lookback)
        x = b[start_reg : i + 1]
        y = a[start_reg : i + 1]

        if x.size >= 2:
            mean_x = x.mean()
            mean_y = y.mean()
            denom = ((x - mean_x) ** 2).sum()
            if denom > 0:
                beta = ((x - mean_x) * (y - mean_y)).sum() / denom
                last_valid_beta = beta
            else:
                beta = last_valid_beta
        else:
            # Not enough points for regression: use last valid
            beta = last_valid_beta

        hedge_ratio[i] = beta

        # Compute spread at time i using hedge ratio computed from data up to i
        spread[i] = a[i] - beta * b[i]

        # Rolling mean/std for spread (use expanding window if not enough points)
        start_z = max(0, i + 1 - zscore_lookback)
        window = spread[start_z : i + 1]

        # window should not contain NaN because we always compute spread for each i
        if window.size > 0:
            m = window.mean()
            s = window.std(ddof=0)
        else:
            m = np.nan
            s = np.nan

        spread_mean[i] = m
        spread_std[i] = s

        if s == 0 or np.isnan(s):
            zscore[i] = 0.0
        else:
            zscore[i] = (spread[i] - m) / s

    return {
        "zscore": zscore,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
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
    Order function for vectorbt flexible multi-asset mode.

    This function returns an order tuple (size, size_type, direction) for the
    asset indicated by c.col at bar index c.i.

    Rules implemented:
    - Entry when zscore > entry_threshold (short A, long B) or zscore < -entry_threshold
      (long A, short B).
    - Exit when zscore crosses zero (sign change) or when |zscore| > stop_threshold.
    - Position sizing: fixed notional per leg (notional_per_leg). Asset A shares
      = notional_per_leg / price_A. Asset B shares = hedge_ratio * asset_A_shares
      and will be sized to be opposite sign to asset A (i.e., target_B = -hedge_ratio * target_A).

    Returns:
        Tuple[size (float), size_type (int), direction (int)]
        - size: absolute quantity (in shares) to trade (positive). If NaN, no order.
        - size_type: integer code indicating type. We use 0 => absolute size (shares).
        - direction: integer code for direction. We use 1 => BUY, 2 => SELL.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Defensive: ensure arrays are numpy and length > i
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Current position for this column (in shares)
    cur_pos = float(getattr(c, "position_now", 0.0))

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If we lack necessary information, do nothing
    if np.isnan(z) or np.isnan(hr) or price_a == 0 or price_b == 0:
        return (np.nan, 0, 0)

    # Compute base share sizing for asset A (absolute shares for the notional)
    a_shares = float(notional_per_leg / price_a) if price_a > 0 else 0.0

    # Determine signals
    entry_short = z > entry_threshold
    entry_long = z < -entry_threshold

    # Detect exit conditions
    cross_zero = False
    if i > 0 and not np.isnan(zscore[i - 1]):
        prev_z = float(zscore[i - 1])
        if prev_z * z < 0:
            cross_zero = True

    stop_loss = abs(z) > stop_threshold

    # Determine target position for asset A. Asset B will be set to -hr * target_A
    target_a = cur_pos  # default keep current
    if stop_loss or cross_zero:
        target_a = 0.0
    else:
        if entry_long:
            target_a = a_shares
        elif entry_short:
            target_a = -a_shares
        else:
            # No new signal: keep current position
            target_a = cur_pos

    # Compute corresponding target for asset B to apply hedge ratio (opposite sign)
    target_b = -hr * target_a

    # Determine which column we are placing order for
    if col == 0:
        target = target_a
        price = price_a
    else:
        target = target_b
        price = price_b

    delta = target - cur_pos

    # If change is negligible, do nothing
    if abs(delta) < 1e-8:
        return (np.nan, 0, 0)

    # Size in absolute shares to trade
    size = float(abs(delta))

    # We use size_type = 0 to indicate absolute size (shares), and
    # direction = 1 for BUY, 2 for SELL.
    size_type = 0
    direction = 1 if delta > 0 else 2

    return (size, size_type, direction)
