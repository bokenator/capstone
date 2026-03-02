import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict


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

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        zscore: Z-score of spread array
        hedge_ratio: Rolling hedge ratio array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)
    """
    i = int(c.i)
    col = int(c.col)
    pos_now = float(getattr(c, 'position_now', 0.0))

    # Basic validation
    if i < 0:
        return (np.nan, 0, 0)

    # Ensure arrays are long enough
    if i >= len(zscore) or i >= len(hedge_ratio) or i >= len(close_a) or i >= len(close_b):
        return (np.nan, 0, 0)

    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If indicator missing, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])

    # Avoid divide by zero
    if price_a == 0 or np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Compute base share sizing (number of shares for Asset A)
    base_shares_a = float(notional_per_leg) / price_a

    # Determine signals
    enter_short = z > entry_threshold  # Short A, Long B
    enter_long = z < -entry_threshold  # Long A, Short B

    # Exit conditions
    stop_loss = abs(z) > stop_threshold

    # Cross-zero detection (including hitting zero)
    cross_zero = False
    if i >= 1:
        prev_z = zscore[i - 1]
        if not np.isnan(prev_z):
            if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
                cross_zero = True

    # If stop loss or crossing zero, close existing positions
    if stop_loss or cross_zero:
        # Close this asset if we have a position
        if pos_now != 0.0:
            # Return an order to close entire position: size = -position_now (shares)
            return (-pos_now, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Entry logic: open or flip positions when thresholds breached
    if enter_short or enter_long:
        # desired sign for Asset A: short => -1, long => +1
        desired_a_sign = -1.0 if enter_short else 1.0

        desired_a = desired_a_sign * base_shares_a
        # Desired B follows hedge ratio: pos_b = -hedge_ratio * pos_a
        desired_b = -hr * desired_a_sign * base_shares_a

        # Select desired based on which column we're processing
        desired = desired_a if col == 0 else desired_b

        # Compute trade size as difference between desired and current
        trade_size = float(desired - pos_now)

        # If trade size is effectively zero, do nothing
        if np.isclose(trade_size, 0.0):
            return (np.nan, 0, 0)

        # Return number of shares to trade (size_type=0 => Amount in shares)
        return (trade_size, 0, 0)

    # If no signal and we already hold position, keep it (no action)
    if pos_now != 0.0:
        return (np.nan, 0, 0)

    # Default: no action
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame or array with 'close' column or close prices for Asset A
        asset_b: DataFrame or array with 'close' column or close prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Helper to extract close prices from various input types
    def _to_close_array(x):
        # If DataFrame with 'close' column
        if isinstance(x, pd.DataFrame):
            if 'close' in x.columns:
                arr = x['close'].values.astype(float)
                return arr
            # If single-column DataFrame
            if x.shape[1] == 1:
                return x.iloc[:, 0].values.astype(float)
            raise ValueError("asset DataFrame must contain a 'close' column")

        # Series
        if isinstance(x, pd.Series):
            return x.values.astype(float)

        # Numpy array or list
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    close_a = _to_close_array(asset_a)
    close_b = _to_close_array(asset_b)

    if close_a.shape[0] != close_b.shape[0]:
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)

    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling OLS regression using only past data (exclude current bar)
    for i in range(n):
        start = max(0, i - hedge_lookback)
        end = i  # exclude current bar to avoid lookahead
        if end - start >= 2:
            x = close_b[start:end]
            y = close_a[start:end]
            # Remove NaNs inside window
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() >= 2:
                try:
                    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                    hedge_ratio[i] = float(slope)
                except Exception:
                    hedge_ratio[i] = np.nan

    # Compute spread using the hedge ratio (no lookahead because hedge_ratio[i] uses past data)
    spread = close_a - hedge_ratio * close_b

    # Rolling z-score: mean and std over past zscore_lookback periods (including current spread)
    # Use min_periods = zscore_lookback to avoid premature signals; will be NaN until enough data
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().to_numpy()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std().to_numpy()

    # Compute z-score safely
    zscore = np.full(n, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        valid = (~np.isnan(spread)) & (~np.isnan(spread_mean)) & (~np.isnan(spread_std))
        denom = spread_std.copy()
        # Avoid division by zero: if std == 0, set zscore to 0.0 (no dispersion)
        zero_std = valid & (denom == 0)
        nonzero_std = valid & (denom != 0)
        zscore[nonzero_std] = (spread[nonzero_std] - spread_mean[nonzero_std]) / denom[nonzero_std]
        zscore[zero_std] = 0.0

    # Ensure output arrays have same length as inputs
    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': hedge_ratio,
        'zscore': zscore,
    }
