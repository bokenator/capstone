"""
Reference Implementation: Complex Strategy (Pairs Trading)
===========================================================

Strategy:
- Rolling OLS regression to compute hedge ratio (lookback 60 periods)
- Spread = Price_A - (hedge_ratio * Price_B)
- Z-score of spread (lookback 20 periods)

Entry conditions:
- Z-score > entry_threshold (2.0): Short A, Long B (spread too high)
- Z-score < -entry_threshold: Long A, Short B (spread too low)

Exit conditions:
- Z-score crosses exit_threshold (0.0): Close both positions
- |Z-score| > stop_threshold (3.0): Stop-loss, close both positions

Function signatures:
    compute_spread_indicators(close_a, close_b, **params) -> dict[str, np.ndarray]
    order_func(c, close_a, close_b, zscore, hedge_ratio, ...) -> tuple
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


# Module-level state for tracking position across bars
_state = {
    "position_type": 0,  # 0=flat, 1=long_spread (long A, short B), -1=short_spread (short A, long B)
}


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> dict[str, np.ndarray]:
    """
    Compute spread indicators for pairs trading strategy.

    Args:
        close_a: Close prices for Asset A (numpy array)
        close_b: Close prices for Asset B (numpy array)
        hedge_lookback: Lookback period for rolling OLS hedge ratio
        zscore_lookback: Lookback period for z-score calculation

    Returns:
        Dict with keys: 'zscore', 'hedge_ratio'
        All values are np.ndarray of same length as input.
    """
    global _state

    # Reset state for fresh backtest
    _state = {
        "position_type": 0,
    }

    n = len(close_a)

    # Compute rolling hedge ratio using OLS regression
    hedge_ratio = np.full(n, np.nan)
    for i in range(hedge_lookback, n):
        # Regress A on B: A = hedge_ratio * B + intercept
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hedge_ratio[i] = slope

    # Compute spread: A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Compute z-score of spread using vectorbt for rolling mean
    spread_series = pd.Series(spread)
    spread_mean = vbt.MA.run(spread_series, window=zscore_lookback).ma.values
    # Note: vectorbt doesn't have a STDEV indicator in VAS, use pandas rolling std
    spread_std = spread_series.rolling(window=zscore_lookback).std().values

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        zscore = np.where(spread_std == 0, np.nan, zscore)

    return {
        "zscore": zscore.astype(float),
        "hedge_ratio": hedge_ratio.astype(float),
    }


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    Uses flexible=True mode where this function is called for each asset at each bar.

    Args:
        c: vectorbt OrderContext with:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        zscore: Z-score of spread array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)

    Returns:
        Tuple of (size, size_type, direction):
        - size: float (np.nan = no action)
        - size_type: int (0=Amount, 1=Value, 2=Percent)
        - direction: int (0=Both, allows long and short)
    """
    global _state

    i = c.i
    col = c.col  # 0 = Asset A, 1 = Asset B
    pos = c.position_now

    # Current values
    z = zscore[i]
    hr = hedge_ratio[i]
    price_a = close_a[i]
    price_b = close_b[i]

    # Skip if indicators not ready
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Determine position type from actual positions
    # This syncs our state with the actual portfolio state
    if col == 0:  # Processing Asset A
        if pos > 0:
            # Long A means we're in "long spread" position
            _state["position_type"] = 1
        elif pos < 0:
            # Short A means we're in "short spread" position
            _state["position_type"] = -1

    position_type = _state["position_type"]

    # Calculate target shares based on notional ($10,000 per leg)
    notional = 10000.0
    shares_a = notional / price_a
    shares_b = (notional / price_b) * abs(hr)

    # Entry logic: no position
    if position_type == 0:
        # Z-score > entry_threshold: spread too high, expect mean reversion down
        # Strategy: Short A, Long B (short the spread)
        if z > entry_threshold:
            _state["position_type"] = -1
            if col == 0:  # Asset A: go short
                return (-shares_a, 0, 0)
            else:  # Asset B: go long
                return (shares_b, 0, 0)

        # Z-score < -entry_threshold: spread too low, expect mean reversion up
        # Strategy: Long A, Short B (long the spread)
        elif z < -entry_threshold:
            _state["position_type"] = 1
            if col == 0:  # Asset A: go long
                return (shares_a, 0, 0)
            else:  # Asset B: go short
                return (-shares_b, 0, 0)

    # Exit logic: have position
    else:
        should_exit = False

        # Exit condition 1: Z-score crosses exit_threshold (mean reversion complete)
        if position_type == -1 and z <= exit_threshold:
            # Was short spread, z-score reverted to normal
            should_exit = True
        elif position_type == 1 and z >= exit_threshold:
            # Was long spread, z-score reverted to normal
            should_exit = True

        # Exit condition 2: Stop-loss (z-score went further against us)
        if abs(z) > stop_threshold:
            should_exit = True

        if should_exit:
            _state["position_type"] = 0
            # Close position for this asset
            if pos != 0:
                return (-pos, 0, 0)

    # No action
    return (np.nan, 0, 0)
