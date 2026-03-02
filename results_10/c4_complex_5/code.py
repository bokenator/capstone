import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Any


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This implementation uses flexible multi-asset mode and returns share
    amount deltas (size_type=0). The function computes desired target
    share exposures for each asset based on z-score signals and a fixed
    notional per leg. It returns the difference between the desired target
    and the current position (c.position_now) so that vectorbt places the
    appropriate order to reach the target.

    Args:
        c: vectorbt OrderContext-like object with attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float, in shares)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A (1D np.ndarray)
        close_b: Close prices for Asset B (1D np.ndarray)
        zscore: Z-score array for the spread (1D np.ndarray)
        hedge_ratio: Rolling hedge ratio array (1D np.ndarray)
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg in dollars

    Returns:
        A tuple (size, size_type, direction):
            - size: float (number of shares to trade, positive=buy, negative=sell)
            - size_type: int, 0=Amount (shares)
            - direction: int, 0=Both (allows long and short)

        If no action is required, returns (np.nan, 0, 0).
    """
    i = int(c.i)
    col = int(getattr(c, 'col', 0))

    # Defensive access to current position (shares). If missing, assume 0.
    pos_now = float(getattr(c, 'position_now', 0.0) or 0.0)

    # Validate indices
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    # Current indicator values
    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if not np.isnan(hedge_ratio[i]) else np.nan

    # If indicators are not available, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Validate prices
    if price_a <= 0 or price_b <= 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute nominal share targets (in shares) for each leg based on fixed notional
    # shares_a: number of Asset A shares that equals notional_per_leg
    # shares_b: scaled by hedge_ratio so that hedge_ratio * shares_b corresponds to shares of A
    # Note: hedge_ratio can be negative; we keep sign to preserve correct directional hedge
    shares_a = float(notional_per_leg / price_a)
    shares_b = float((notional_per_leg / price_b) * hr)

    # Determine previous zscore for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and not np.isnan(zscore[i - 1]) else np.nan

    # Determine signals
    # 1) Stop-loss has highest priority
    if abs(z) > stop_threshold:
        desired_a = 0.0
        desired_b = 0.0
    else:
        # 2) Exit when z-score crosses exit_threshold (usually 0.0)
        crossed_exit = False
        if not np.isnan(prev_z):
            # Crossing from above to <= exit_threshold
            if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < exit_threshold and z >= exit_threshold):
                crossed_exit = True

        if crossed_exit:
            desired_a = 0.0
            desired_b = 0.0
        else:
            # 3) Entries
            if z > entry_threshold:
                # Short A, Long B
                desired_a = -shares_a
                desired_b = +shares_b
            elif z < -entry_threshold:
                # Long A, Short B
                desired_a = +shares_a
                desired_b = -shares_b
            else:
                # No signal -> no trading action
                return (np.nan, 0, 0)

    # Select desired target for this column
    desired_target = desired_a if col == 0 else desired_b

    # Compute order size as difference between desired target and current position (shares)
    order_size = float(desired_target - pos_now)

    # If the required change is negligible, do nothing
    if np.isclose(order_size, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    # Return number of shares to trade (size_type=0 => amount/shares), allow both long & short
    return (order_size, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible and accepts either pandas DataFrame/Series
    inputs with a 'close' column or plain 1D numpy arrays of close prices.

    Returns a dictionary containing:
        - 'close_a': np.ndarray of Asset A closes
        - 'close_b': np.ndarray of Asset B closes
        - 'hedge_ratio': np.ndarray of rolling OLS slopes (NaN where unavailable)
        - 'zscore': np.ndarray of spread z-score (NaN where unavailable)
    """

    def _extract_close(x: Any) -> np.ndarray:
        # Accept numpy arrays
        if isinstance(x, np.ndarray):
            return x.astype(float)
        # pandas Series
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # pandas DataFrame: use 'close' column only (per DATA_SCHEMA)
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                return x['close'].values.astype(float)
            # If a single-column DataFrame without 'close', take first column
            if x.shape[1] == 1:
                return x.iloc[:, 0].values.astype(float)
        # As a fallback, try to convert
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.astype(float)
        raise ValueError("asset input must be a 1D array/Series or DataFrame with a 'close' column")

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS for hedge ratio: slope of regression of A ~ B over lookback window
    # Place computed slope at the index i (using previous 'hedge_lookback' bars as in prompt example)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be >= 2")

    for i in range(hedge_lookback, n):
        y = close_a[i - hedge_lookback:i]
        x = close_b[i - hedge_lookback:i]

        # Skip if there are NaNs in the regression window
        if np.isnan(x).any() or np.isnan(y).any():
            hedge_ratio[i] = np.nan
            continue

        # If x is constant, slope is undefined -> skip
        if np.isclose(x.std(ddof=0), 0.0):
            hedge_ratio[i] = np.nan
            continue

        # Compute slope via linregress
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Compute spread: A - hedge_ratio * B
    spread = np.full(n, np.nan, dtype=float)
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Compute rolling mean and std of spread using pandas (to handle NaNs reliably)
    spread_series = pd.Series(spread)
    # Require full window to produce statistic (min_periods = zscore_lookback)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score and guard against divide-by-zero
    zscore = np.full(n, np.nan, dtype=float)
    valid_z = ~np.isnan(spread) & ~np.isnan(spread_mean) & (spread_std > 0)
    zscore[valid_z] = (spread[valid_z] - spread_mean[valid_z]) / spread_std[valid_z]

    return {
        'close_a': close_a,
        'close_b': close_b,
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
