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
    Generate orders for a pairs trading strategy in flexible (multi-asset) mode.

    The function computes target positions in shares for Asset A and Asset B based
    on z-score signals and the rolling hedge ratio, then returns the order size
    (difference between target and current position) for the asset represented by
    the provided OrderContext `c`.

    Size type: returns number of shares (size_type=0). Direction=0 (both allowed).

    Args:
        c: Order context (has attributes i, col, position_now, cash_now)
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array of z-score values for the spread
        hedge_ratio: 1D array of rolling hedge ratio (slope) values
        entry_threshold: z-score threshold to open positions (e.g., 2.0)
        exit_threshold: z-score threshold used for exit crossing (e.g., 0.0)
        stop_threshold: absolute z-score threshold for stop-loss (e.g., 3.0)
        notional_per_leg: fixed notional used to size the A leg (shares_a = notional/price_a)

    Returns:
        (size, size_type, direction)
        - size: number of shares to trade (positive=buy, negative=sell)
        - size_type: 0 (Amount / shares)
        - direction: 0 (Both)

    Notes:
        - Uses target sizing: shares_a = notional_per_leg / price_a
          shares_b = shares_a * hedge_ratio
        - Target positions (in shares) when signal triggers:
            z > entry_threshold  -> short A, long B (target_A = -shares_a, target_B = +shares_b)
            z < -entry_threshold -> long A, short B (target_A = +shares_a, target_B = -shares_b)
        - Exit when z crosses exit_threshold (usually 0) or when |z| > stop_threshold
        - If hedge_ratio is NaN at the bar, no new trades are generated
    """
    i = int(getattr(c, "i"))
    col = int(getattr(c, "col"))
    pos = getattr(c, "position_now", 0.0)

    # Basic bounds checks
    n = len(zscore)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    z = zscore[i]
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = close_a[i] if i < len(close_a) else np.nan
    price_b = close_b[i] if i < len(close_b) else np.nan

    # Validate prices
    if np.isnan(price_a) or np.isnan(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Hedge ratio must be available to size pair correctly
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan
    if np.isnan(hr):
        # Do not trade until a valid hedge ratio is available
        return (np.nan, 0, 0)

    # Compute share sizing
    shares_a = float(notional_per_leg) / float(price_a)
    shares_b = shares_a * float(hr)

    # Determine signals
    stop_loss = np.abs(z) > float(stop_threshold)

    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_to_exit = False
    if not np.isnan(prev_z):
        # Cross if previous z was on opposite side and current has reached/passed exit_threshold
        if (prev_z > 0 and z <= float(exit_threshold)) or (prev_z < 0 and z >= float(exit_threshold)):
            crossed_to_exit = True

    enter_short_spread = z > float(entry_threshold)
    enter_long_spread = z < -float(entry_threshold)

    # Determine target (in shares) for this asset
    target_shares = None

    if stop_loss or crossed_to_exit:
        # Close both legs
        target_shares = 0.0
    elif enter_short_spread:
        # Short A, Long B (use hedge ratio to size B)
        if col == 0:
            target_shares = -shares_a
        else:
            target_shares = +shares_b
    elif enter_long_spread:
        # Long A, Short B
        if col == 0:
            target_shares = +shares_a
        else:
            target_shares = -shares_b
    else:
        # No signal change -> no action
        return (np.nan, 0, 0)

    # Compute order amount (shares) to move from current position to target
    # Ensure pos is numeric
    try:
        pos_now = float(pos)
    except Exception:
        pos_now = 0.0

    order_shares = float(target_shares) - pos_now

    # Small tolerance to avoid tiny orders due to floating point
    if np.isclose(order_shares, 0.0, atol=1e-8):
        return (np.nan, 0, 0)

    return (order_shares, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, pd.Series, np.ndarray],
    asset_b: Union[pd.DataFrame, pd.Series, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread for a pair of assets.

    Args:
        asset_a: DataFrame with 'close' column, Series or 1D numpy array of closes for Asset A
        asset_b: DataFrame with 'close' column, Series or 1D numpy array of closes for Asset B
        hedge_lookback: lookback window (in bars) to compute rolling OLS hedge ratio
        zscore_lookback: lookback window (in bars) to compute rolling mean/std of spread

    Returns:
        dict with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'
        Each value is a 1D numpy array of length n (same as input closes)

    Notes:
        - The function only reads the 'close' column if DataFrames are provided (per DATA_SCHEMA)
        - Rolling regression requires full non-NaN windows; otherwise hedge_ratio remains NaN
        - Rolling z-score uses pandas rolling with min_periods equal to the lookback
    """
    # Extract close arrays depending on input type
    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            arr = x
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, pd.DataFrame):
            if "close" not in x.columns:
                raise KeyError("DataFrame input must contain 'close' column per DATA_SCHEMA")
            arr = x["close"].values
        else:
            # Try to coerce
            arr = np.asarray(x)
        # Ensure 1D float array
        arr = np.asarray(arr, dtype="float64")
        if arr.ndim != 1:
            raise ValueError("Close price input must be a 1D array/Series/DataFrame column")
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError("Asset A and Asset B must have the same length")

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError("hedge_lookback must be at least 2")
    if zscore_lookback < 1:
        raise ValueError("zscore_lookback must be at least 1")

    # Rolling OLS for hedge ratio (slope of regression of A ~ B)
    hedge_ratio = np.full(n, np.nan, dtype="float64")

    for i in range(hedge_lookback, n + 1):
        # window is [i-hedge_lookback, i)
        start = i - hedge_lookback
        end = i
        y = close_a[start:end]
        x = close_b[start:end]
        # Require no NaNs in the regression window
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        # Require variation in x
        if np.allclose(x, x[0]):
            # no variation -> slope is undefined
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        except Exception:
            continue
        hedge_ratio[end - 1] = float(slope)

    # Compute spread using hedge_ratio where available
    spread = np.full(n, np.nan, dtype="float64")
    valid_hr = ~np.isnan(hedge_ratio)
    spread[valid_hr] = close_a[valid_hr] - hedge_ratio[valid_hr] * close_b[valid_hr]

    # Rolling mean and std for z-score
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread_series.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0)

    # Compute z-score safely
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (spread - spread_mean.values) / spread_std.values

    # Force zscore to NaN where spread is NaN or std is zero/NaN
    invalid_mask = np.isnan(spread) | np.isnan(spread_std) | (spread_std == 0)
    zscore = np.where(invalid_mask, np.nan, zscore)

    return {
        "close_a": close_a,
        "close_b": close_b,
        "hedge_ratio": hedge_ratio,
        "zscore": np.asarray(zscore, dtype="float64"),
    }
