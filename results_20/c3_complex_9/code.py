# Pairs trading: compute_spread_indicators and order_func

from typing import Dict, Union, Any
import numpy as np
import pandas as pd


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series],
    close_b: Union[np.ndarray, pd.Series],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pairs trading strategy.

    Args:
        close_a: Prices for asset A (array-like)
        close_b: Prices for asset B (array-like)
        hedge_lookback: Lookback window for rolling OLS (uses expanding window when fewer bars available)
        zscore_lookback: Lookback window for rolling mean/std of spread (uses expanding window when fewer bars available)

    Returns:
        Dictionary with keys:
            - 'hedge_ratio': np.ndarray of hedge ratios (slope of OLS A ~ B)
            - 'zscore': np.ndarray of z-scores of the spread
            - 'spread': np.ndarray of spread values
    
    Notes:
        - Uses only past and present data at each index (no lookahead).
        - For early bars where full lookback is not available, an expanding-window estimate is used.
        - Numerical stability handled for low-variance windows.
    """
    # Convert to numpy arrays of float
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Small epsilon for numerical checks
    EPS = 1e-12

    for i in range(n):
        # Rolling/expanding window for hedge ratio (use available history up to hedge_lookback)
        start_h = max(0, i - hedge_lookback + 1)
        xb = b[start_h : i + 1]
        ya = a[start_h : i + 1]

        mask = np.isfinite(xb) & np.isfinite(ya)
        x = xb[mask]
        y = ya[mask]

        hr = np.nan
        if x.size == 0:
            hr = np.nan
        elif x.size == 1:
            # If single point, estimate slope directly (avoid divide by zero)
            if abs(x[0]) < EPS:
                hr = 0.0
            else:
                hr = float(y[0]) / float(x[0])
        else:
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom <= EPS:
                hr = 0.0
            else:
                hr = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

        hedge_ratio[i] = hr

        # Spread at time i (use current hedge ratio estimate)
        if np.isfinite(hr) and np.isfinite(a[i]) and np.isfinite(b[i]):
            spread[i] = a[i] - hr * b[i]
        else:
            spread[i] = np.nan

        # Rolling mean/std for spread -> z-score (expanding when fewer bars)
        start_z = max(0, i - zscore_lookback + 1)
        window = spread[start_z : i + 1]
        wmask = np.isfinite(window)
        w = window[wmask]

        if w.size == 0 or not np.isfinite(spread[i]):
            zscore[i] = np.nan
        else:
            mean_w = float(w.mean())
            std_w = float(w.std(ddof=0))
            if std_w < EPS:
                # If zero variance, set z-score to 0 to avoid infinite values
                zscore[i] = 0.0
            else:
                zscore[i] = (spread[i] - mean_w) / std_w

    return {
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
        "spread": spread,
    }


# Order function for vectorbt flexible mode
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
) -> tuple:
    """
    Determines per-asset orders for a pairs trading strategy.

    Arguments defined to match vectorbt.from_order_func flexible mode wrapper in the test harness.

    Returns a tuple (size, size_type, direction) where:
      - size: float (number of units to trade) or np.nan for no order
      - size_type: int (0 -> absolute units)
      - direction: int (1 -> long/buy, 2 -> short/sell)

    The function uses only past and present information (c.i index) so no lookahead occurs.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Retrieve current values
    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    pos_now = float(getattr(c, "position_now", 0.0))

    # Treat very small positions as zero
    if abs(pos_now) < 1e-12:
        pos_now = 0.0

    SIZE_TYPE_SIZE = 0  # interpret size as number of units
    DIRECTION_LONG = 1
    DIRECTION_SHORT = 2

    # Helper: no order
    def no_order():
        return (np.nan, SIZE_TYPE_SIZE, 0)

    # Validate existence of required data
    if np.isnan(z) or np.isnan(hr) or np.isnan(price_a) or np.isnan(price_b):
        return no_order()

    # Base unit size for asset A (so that asset A notional ~= notional_per_leg)
    # If price is 0 or invalid, return no order
    if price_a <= 0 or price_b <= 0:
        return no_order()

    base_size_a = float(notional_per_leg / price_a)

    # Determine actions
    # 1) Stop-loss: if |z| > stop_threshold -> close any open positions
    if abs(z) > float(stop_threshold):
        if pos_now != 0.0:
            # Close by placing an opposite order equal to current position magnitude
            size = abs(pos_now)
            # If currently long (pos_now>0) -> sell (short direction), else buy
            direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
            return (size, SIZE_TYPE_SIZE, int(direction))
        return no_order()

    # 2) Exit on mean reversion: z-score crosses zero -> if there is a position, close it
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan
    if np.isfinite(prev_z) and prev_z * z < 0:
        if pos_now != 0.0:
            size = abs(pos_now)
            direction = DIRECTION_SHORT if pos_now > 0 else DIRECTION_LONG
            return (size, SIZE_TYPE_SIZE, int(direction))
        return no_order()

    # 3) Entry conditions
    # Long A / Short B when z < -entry_threshold
    if z < -float(entry_threshold):
        if col == 0:
            # Asset A: go long if flat
            if pos_now == 0.0:
                size = base_size_a
                return (size, SIZE_TYPE_SIZE, int(DIRECTION_LONG))
            else:
                return no_order()
        else:
            # Asset B: go short hedge_ratio * base_size_a
            if pos_now == 0.0:
                size = abs(hr) * base_size_a
                return (size, SIZE_TYPE_SIZE, int(DIRECTION_SHORT))
            else:
                return no_order()

    # Short A / Long B when z > entry_threshold
    if z > float(entry_threshold):
        if col == 0:
            # Asset A: go short
            if pos_now == 0.0:
                size = base_size_a
                return (size, SIZE_TYPE_SIZE, int(DIRECTION_SHORT))
            else:
                return no_order()
        else:
            # Asset B: go long hedge_ratio * base_size_a
            if pos_now == 0.0:
                size = abs(hr) * base_size_a
                return (size, SIZE_TYPE_SIZE, int(DIRECTION_LONG))
            else:
                return no_order()

    # Otherwise, no order
    return no_order()
