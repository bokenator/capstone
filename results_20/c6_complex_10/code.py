# Pairs trading indicator and order function implementation
# Implements rolling OLS hedge ratio, spread, z-score, and a flexible order function
# that sizes positions in amount (shares) so that asset B is scaled by the hedge ratio.

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
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs trading strategy.

    Supports 1D and 2D inputs (multiple columns). If inputs are 1D, outputs are 1D.
    If inputs are 2D (shape (n, k)), outputs are 2D with the same shape. If k == 1,
    the outputs are squeezed to 1D for convenience.

    Args:
        close_a: Prices of asset A (array-like) shape (n,) or (n, k)
        close_b: Prices of asset B (array-like) shape (n,) or (n, k)
        hedge_lookback: Lookback window for rolling OLS (max window). For early bars,
                       an expanding window (using all available history up to t) is used.
        zscore_lookback: Lookback for rolling mean/std of the spread.

    Returns:
        Dict with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (slope of regressing A on B) at each t
            - 'zscore': np.ndarray of z-scores of the spread at each t
            - 'spread': np.ndarray of spread values

    Notes:
        - All calculations use only data up to the current index (no lookahead).
        - For degenerate windows (zero variance in B), hedge_ratio is set to 0.
        - Rolling mean/std use min_periods=1 to avoid NaNs; if std is zero, z-score is set to 0.
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.ndim > 2 or b.ndim > 2:
        raise ValueError("close_a and close_b must be 1D or 2D arrays")

    # Normalize inputs to 2D for unified processing: shape (n, cols)
    if a.ndim == 1 and b.ndim == 1:
        a2 = a.reshape(-1, 1)
        b2 = b.reshape(-1, 1)
    else:
        # If one is 1D and other is 2D, attempt to broadcast 1D to 2D by repeating columns
        if a.ndim == 1 and b.ndim == 2:
            a2 = np.repeat(a.reshape(-1, 1), b.shape[1], axis=1)
            b2 = b
        elif b.ndim == 1 and a.ndim == 2:
            b2 = np.repeat(b.reshape(-1, 1), a.shape[1], axis=1)
            a2 = a
        else:
            a2 = a
            b2 = b

    if a2.shape != b2.shape:
        raise ValueError("close_a and close_b must have compatible shapes")

    n, cols = a2.shape

    hedge_ratio = np.full((n, cols), np.nan, dtype=float)
    eps = 1e-12

    # Compute rolling OLS for each column independently
    for col in range(cols):
        col_a = a2[:, col]
        col_b = b2[:, col]
        for t in range(n):
            start = 0 if hedge_lookback is None else max(0, t - hedge_lookback + 1)
            x = col_b[start : t + 1]
            y = col_a[start : t + 1]
            if x.size < 2:
                hedge_ratio[t, col] = 0.0
                continue
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom <= eps:
                hedge_ratio[t, col] = 0.0
            else:
                numer = np.sum((x - x_mean) * (y - y_mean))
                hedge_ratio[t, col] = numer / denom

    # Compute spread: elementwise
    spread = a2 - hedge_ratio * b2

    # Rolling mean/std per column using pandas DataFrame for convenience
    df_spread = pd.DataFrame(spread)
    roll_mean = df_spread.rolling(window=zscore_lookback, min_periods=1).mean()
    roll_std = df_spread.rolling(window=zscore_lookback, min_periods=1).std(ddof=0)

    zscore = np.full_like(spread, 0.0, dtype=float)
    for col in range(cols):
        for t in range(n):
            std_t = float(roll_std.iloc[t, col])
            if not np.isfinite(std_t) or std_t <= eps:
                zscore[t, col] = 0.0
            else:
                zscore[t, col] = float((spread[t, col] - float(roll_mean.iloc[t, col])) / std_t)

    # Squeeze back to 1D if single column for convenience
    if cols == 1:
        return {
            "hedge_ratio": hedge_ratio.ravel(),
            "zscore": zscore.ravel(),
            "spread": spread.ravel(),
        }

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
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
    Flexible order function for pairs trading.

    This function is designed to be used in a flexible multi-asset order flow where it
    will be called once per column per bar. It returns a tuple (size, size_type, direction).

    SizeType uses Amount (0) so `size` is number of shares. Direction uses 0=LongOnly, 1=ShortOnly, 2=Both.

    Logic summary:
      - Entry when |zscore| > entry_threshold:
          z > +entry -> Short A, Long B
          z < -entry -> Long A, Short B
        Sizes (in shares):
          shares_a = notional_per_leg / price_a
          shares_b = abs(hedge_ratio) * shares_a

      - Exit when zscore crosses exit_threshold (0.0) OR |zscore| > stop_threshold (stop-loss):
          Close both legs (issue closing orders equal to current position magnitude) using Direction.Both.

    Returns (np.nan, 0, 2) if no order should be placed for this column at current bar.
    """
    # Defensive conversions
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    position_now = float(getattr(c, "position_now", 0.0))

    size_type_amount = 0  # SizeType.Amount
    dir_long_only = 0  # Direction.LongOnly
    dir_short_only = 1  # Direction.ShortOnly
    dir_both = 2  # Direction.Both

    # Safety checks
    if i < 0:
        return (np.nan, size_type_amount, dir_both)

    # Support 2D zscore/hedge inputs: if provided as 1D, treat as 1-column
    if zscore.ndim == 1:
        z_t = float(zscore[i]) if i < zscore.shape[0] else 0.0
    else:
        z_t = float(zscore[i, col]) if i < zscore.shape[0] and col < zscore.shape[1] else 0.0

    if hedge_ratio.ndim == 1:
        h_t = float(hedge_ratio[i]) if i < hedge_ratio.shape[0] else 0.0
    else:
        h_t = float(hedge_ratio[i, col]) if i < hedge_ratio.shape[0] and col < hedge_ratio.shape[1] else 0.0

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    eps = 1e-12

    # --- Closing logic: if we are currently in a position and exit/stop conditions met ---
    crossed_to_exit = False
    if i > 0:
        if zscore.ndim == 1:
            prev_z = float(zscore[i - 1])
        else:
            prev_z = float(zscore[i - 1, col]) if col < zscore.shape[1] else 0.0
        if (prev_z > exit_threshold and z_t <= exit_threshold) or (prev_z < exit_threshold and z_t >= exit_threshold):
            crossed_to_exit = True

    stop_loss = abs(z_t) > stop_threshold

    if abs(position_now) > eps:
        if crossed_to_exit or stop_loss:
            close_size = abs(position_now)
            if close_size <= eps:
                return (np.nan, size_type_amount, dir_both)
            # Use Direction.Both for closing to avoid direction-only conflicts
            return (float(close_size), size_type_amount, dir_both)
        else:
            return (np.nan, size_type_amount, dir_both)

    # --- Entry logic: only when currently flat ---
    if abs(position_now) <= eps:
        # Validate price and hedge ratio
        if price <= 0 or not np.isfinite(price) or not np.isfinite(h_t):
            return (np.nan, size_type_amount, dir_both)

        shares_a = notional_per_leg / float(close_a[i])
        shares_b = abs(h_t) * shares_a

        if z_t > entry_threshold:
            # Short A, Long B
            if col == 0:
                return (float(shares_a), size_type_amount, dir_short_only)
            else:
                return (float(shares_b), size_type_amount, dir_long_only)

        if z_t < -entry_threshold:
            # Long A, Short B
            if col == 0:
                return (float(shares_a), size_type_amount, dir_long_only)
            else:
                return (float(shares_b), size_type_amount, dir_short_only)

    return (np.nan, size_type_amount, dir_both)
