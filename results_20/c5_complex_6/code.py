import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Union


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
    col = int(getattr(c, "col", 0))  # 0 = Asset A, 1 = Asset B
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validation
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    hr = hedge_ratio[i]

    # If any key value is NaN, skip
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    price_a = close_a[i]
    price_b = close_b[i]

    # Avoid divide by zero or invalid prices
    if price_a is None or price_b is None:
        return (np.nan, 0, 0)
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base share sizing: number of shares of A based on fixed notional
    shares_a = float(notional_per_leg) / float(price_a)

    # Hedge ratio scales B shares relative to A shares
    shares_b = shares_a * float(hr)

    # Determine pair-level signals
    # Check for stop-loss first (must close positions)
    if abs(z) > stop_threshold:
        # Close any open position for this asset
        if pos_now != 0.0:
            return (-pos_now, 0, 0)
        else:
            return (np.nan, 0, 0)

    # Check for exit on mean reversion (z-score crossing zero)
    # Need previous z-score to detect crossing
    if pos_now != 0.0 and i > 0:
        z_prev = zscore[i - 1]
        if not np.isnan(z_prev):
            # If sign changed or exact crossing through zero, close
            if z_prev * z <= 0:
                return (-pos_now, 0, 0)

    # Entry conditions only when currently flat
    if pos_now == 0.0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Asset A: short shares_a
                return (-shares_a, 0, 0)
            else:
                # Asset B: long shares_b
                return (shares_b, 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Asset A: long shares_a
                return (shares_a, 0, 0)
            else:
                # Asset B: short shares_b
                return (-shares_b, 0, 0)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, np.ndarray],
    asset_b: Union[pd.DataFrame, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    This function is flexible and accepts either:
      - Two pandas DataFrames with a 'close' column each (asset_a, asset_b), or
      - Two numpy arrays of close prices (1D).

    Args:
        asset_a: DataFrame or 1D array with close prices for Asset A
        asset_b: DataFrame or 1D array with close prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close price arrays from inputs
    def _extract_close(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if "close" in x.columns:
                arr = x["close"].values.astype(float)
            else:
                # If dataframe of raw prices (e.g., single column), take first column
                arr = x.iloc[:, 0].values.astype(float)
        elif isinstance(x, (np.ndarray, pd.Series)):
            arr = np.asarray(x).astype(float)
            if arr.ndim > 1:
                # Flatten if shape is (n,1)
                arr = arr.ravel()
        else:
            # Fallback: try to convert
            arr = np.asarray(x).astype(float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset price arrays must have the same length")

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Compute rolling hedge ratio using OLS on a trailing window ending at i (inclusive)
    # We allow smaller windows at the beginning (min 2 points) to avoid long warmup NaNs
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start : i + 1]
        y = close_a[start : i + 1]
        # Remove NaNs from the window
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            hedge_ratio[i] = np.nan
            continue
        x_clean = x[mask]
        y_clean = y[mask]
        # If all x are constant, slope is 0
        if np.allclose(x_clean, x_clean[0]):
            # Avoid degenerate regression
            hedge_ratio[i] = 0.0
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        hedge_ratio[i] = float(slope)

    # Compute spread using the hedge ratio at each time (no lookahead)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score. Use min_periods=1 to avoid NaNs after warmup
    spread_ser = pd.Series(spread)
    spread_mean = spread_ser.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use ddof=0 for population std to avoid NaNs with single sample
    spread_std = spread_ser.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Avoid division by zero: when std is zero, set zscore to 0
    zscore = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(spread[i]) or np.isnan(spread_mean[i]) or np.isnan(spread_std[i]):
            zscore[i] = np.nan
            continue
        if spread_std[i] == 0:
            zscore[i] = 0.0
        else:
            zscore[i] = (spread[i] - spread_mean[i]) / spread_std[i]

    # Final cleanup: ensure arrays are 1D numpy arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)
    zscore = np.asarray(zscore, dtype=float)

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": zscore,
    }
