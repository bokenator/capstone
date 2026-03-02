# Pairs trading strategy implementation for vectorbt
# Exports:
# - compute_spread_indicators
# - order_func

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd


def compute_spread_indicators(
    prices_a: Any,
    prices_b: Any,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS slope), spread, and z-score for a pairs trading strategy.

    Args:
        prices_a: Array-like of prices for asset A (numpy array or pandas Series).
        prices_b: Array-like of prices for asset B (numpy array or pandas Series).
        hedge_lookback: Lookback window for rolling OLS (will use smaller window at start to avoid NaNs).
        zscore_lookback: Lookback window for rolling mean/std of the spread (will use smaller window at start).

    Returns:
        dict with keys:
            - 'hedge_ratio': numpy array of rolling hedge ratios (slope)
            - 'spread': numpy array of spread values (Price_A - hedge_ratio * Price_B)
            - 'zscore': numpy array of z-scores of the spread

    Notes:
        - Computations are performed using only historical data up to and including the current index
          (no lookahead).
        - For early indices where the full lookback is not available, the function uses the available
          data (i.e., expanding window behaviour) to avoid NaNs after a small warmup.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("prices_a and prices_b must have the same shape")

    n = a.shape[0]

    hedge_ratio = np.zeros(n, dtype=float)
    spread = np.zeros(n, dtype=float)
    zscore = np.zeros(n, dtype=float)

    # Small epsilon to avoid division by zero
    EPS = 1e-12

    for t in range(n):
        # Rolling OLS regression y = a + slope * x using available window up to t (inclusive)
        w = min(hedge_lookback, t + 1)
        x = b[t - w + 1 : t + 1]
        y = a[t - w + 1 : t + 1]

        # Compute slope = cov(x, y) / var(x)
        mean_x = x.mean()
        mean_y = y.mean()
        denom = np.sum((x - mean_x) ** 2)
        if denom < EPS:
            slope = 0.0
        else:
            slope = np.sum((x - mean_x) * (y - mean_y)) / denom

        hedge_ratio[t] = slope

        # Spread using the current slope
        spread[t] = a[t] - slope * b[t]

        # Rolling mean/std of spread for z-score (use available window)
        w2 = min(zscore_lookback, t + 1)
        sp_win = spread[t - w2 + 1 : t + 1]
        mu = sp_win.mean()
        sigma = sp_win.std(ddof=0)
        if sigma < EPS:
            z = 0.0
        else:
            z = (spread[t] - mu) / sigma
        zscore[t] = z

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
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
    Order function for flexible multi-asset vectorbt backtest (pairs trading).

    This function follows the required signature and returns a simple Python tuple
    (size, size_type, direction). It does not use numba and does not return numba objects.

    Strategy logic implemented:
    - Entry: |zscore| > entry_threshold -> open a pair trade
        - zscore > entry: SHORT A, LONG B
        - zscore < -entry: LONG A, SHORT B
      Positions are sized so that Asset A amount (in units) = notional_per_leg / price_A,
      and Asset B units = hedge_ratio * Asset_A_units (i.e., hedge ratio applied to units).
      We use SizeType.Amount (units) for entries.

    - Exit: zscore crosses 0.0 OR |zscore| > stop_threshold -> close both positions
      For exits we use SizeType.TargetAmount and set target to 0 (close position).

    Notes on returned integers (do not import enums to remain numba-free):
    - SizeType: Amount=0, Value=1, Percent=2, TargetAmount=3, TargetValue=4, TargetPercent=5
    - TradeDirection/Order direction: Long=0, Short=1, Both=2 (Order default is 2)

    Args:
        c: Context object provided by vectorbt. Expect attributes .i (index) and .col (column 0 or 1),
           and .position_now (current position for the column) or similar provided by the test wrapper.
        close_a: numpy array of close prices for asset A.
        close_b: numpy array of close prices for asset B.
        zscore: numpy array of z-scores computed by compute_spread_indicators.
        hedge_ratio: numpy array of hedge ratios computed by compute_spread_indicators.
        entry_threshold: threshold for entry (default 2.0)
        exit_threshold: threshold for exit crossing (default 0.0)
        stop_threshold: stop loss threshold (default 3.0)
        notional_per_leg: fixed notional per leg (USD) used to compute base units for asset A.

    Returns:
        Tuple of (size, size_type, direction). If no order, return (np.nan, 0, 0).
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Retrieve price and current position (position_now provided by wrapper)
    price = float(close_a[i]) if col == 0 else float(close_b[i])
    position_now = float(getattr(c, "position_now", 0.0))

    # Tolerate tiny positions as zero
    if abs(position_now) < 1e-8:
        position_now = 0.0

    # Safeguard indices
    if i < 0 or i >= len(zscore) or i >= len(hedge_ratio):
        return (np.nan, 0, 0)

    zs = float(zscore[i])
    hr = float(hedge_ratio[i])

    # Base units for asset A (units = notional / price)
    # Guard against zero price
    if price == 0 or not np.isfinite(price):
        return (np.nan, 0, 0)

    # Constants for SizeType and Direction (use ints to avoid importing enums)
    SIZE_AMOUNT = 0
    SIZE_VALUE = 1
    SIZE_TARGET_AMOUNT = 3
    DIR_LONG = 0
    DIR_SHORT = 1
    DIR_BOTH = 2

    # Use asset A price for sizing base units
    price_a = float(close_a[i])
    if price_a <= 0 or not np.isfinite(price_a):
        return (np.nan, 0, 0)

    base_units = float(notional_per_leg) / price_a

    # Determine desired action for this column
    # Entry logic: only enter when currently flat (position_now == 0)
    is_flat = position_now == 0.0

    # Exit logic: when in a trade, check stop-loss or z-score crossing 0
    crossed_zero = False
    if i > 0:
        prev_zs = float(zscore[i - 1])
        # Consider crossing if signs differ (treat sign(0) carefully)
        if np.sign(prev_zs) != np.sign(zs):
            crossed_zero = True
        # If previous was exactly zero and now also zero, treat as crossed
        if prev_zs == 0 and zs == 0:
            crossed_zero = True

    # Stop loss condition
    stop_loss = abs(zs) > float(stop_threshold)

    # Exit condition satisfied?
    exit_now = stop_loss or crossed_zero or (abs(zs) <= float(exit_threshold))

    # Column-specific decision
    if col == 0:
        # Asset A
        if is_flat:
            # Entry decisions
            if zs > float(entry_threshold):
                # SHORT A (open short)
                size = base_units
                return (float(size), SIZE_AMOUNT, DIR_SHORT)
            elif zs < -float(entry_threshold):
                # LONG A
                size = base_units
                return (float(size), SIZE_AMOUNT, DIR_LONG)
            else:
                return (np.nan, 0, 0)
        else:
            # Not flat -> consider exit
            if exit_now:
                # Use target amount 0 to close position
                return (0.0, SIZE_TARGET_AMOUNT, DIR_BOTH)
            else:
                return (np.nan, 0, 0)
    else:
        # Asset B
        if is_flat:
            # Need to compute B units so that pos_b ~= hedge_ratio * base_units
            b_units = abs(hr) * base_units

            if zs > float(entry_threshold):
                # For zs > entry: SHORT A, LONG B (if hr >=0). If hr<0 sign flips.
                # Decide B direction so that pos_b == -hr * pos_a holds.
                # pos_a will be negative (short) -> pos_b should be hr * base_units.
                # Determine direction for B based on hr sign relative to A direction.
                # A direction = DIR_SHORT
                dA = DIR_SHORT
                if hr < 0:
                    dB = dA  # same sign as A
                else:
                    dB = 1 - dA  # opposite sign

                return (float(b_units), SIZE_AMOUNT, int(dB))
            elif zs < -float(entry_threshold):
                # For zs < -entry: LONG A, SHORT B
                dA = DIR_LONG
                if hr < 0:
                    dB = dA
                else:
                    dB = 1 - dA
                return (float(b_units), SIZE_AMOUNT, int(dB))
            else:
                return (np.nan, 0, 0)
        else:
            # Not flat -> consider exit
            if exit_now:
                return (0.0, SIZE_TARGET_AMOUNT, DIR_BOTH)
            else:
                return (np.nan, 0, 0)
