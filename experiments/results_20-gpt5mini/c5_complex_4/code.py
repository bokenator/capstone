import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Any, Dict, Tuple, Union


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
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    Parameters
    ----------
    c : OrderContext-like object
        Must expose: i (int), col (int: 0=A,1=B), position_now (float), cash_now (float)
    close_a, close_b : np.ndarray
        Close price arrays for Asset A and B
    zscore : np.ndarray
        Z-score array for the spread
    hedge_ratio : np.ndarray
        Rolling hedge ratio array (slope of regressing A on B), aligned so that
        hedge_ratio[i] is computed using data strictly before index i
    entry_threshold, exit_threshold, stop_threshold : float
        Thresholds described in the strategy
    notional_per_leg : float
        Fixed notional per leg in dollars

    Returns
    -------
    Tuple[size, size_type, direction]
        size: float (positive buy, negative sell), size_type: 0=Amount(shares), 1=Value($), 2=Percent
        direction: 0=Both, 1=LongOnly, 2=ShortOnly
    """
    i = int(c.i)
    col = int(c.col)
    pos = float(c.position_now) if hasattr(c, 'position_now') else 0.0

    # Basic guards
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    z = zscore[i]
    # If z-score is not available, do nothing
    if np.isnan(z):
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan

    if np.isnan(price_a) or np.isnan(price_b):
        return (np.nan, 0, 0)

    # Hedge ratio - fallback to 1.0 if not available to avoid crashing the order logic
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio) and not np.isnan(hedge_ratio[i])) else 1.0

    # Determine share sizing: we maintain number-of-shares relationship such that
    # shares_b = hedge_ratio * shares_a. Fixed notional per leg is used to size shares_a.
    # shares_a represents absolute (positive) quantity per leg before sign.
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    shares_a = float(notional_per_leg / price_a)
    shares_b = float(shares_a * hr)

    # Numerical tolerance for zero-checks
    if np.isclose(pos, 0.0, atol=1e-12):
        in_position = False
    else:
        in_position = True

    # ENTRY logic (only enter if currently flat for this asset)
    if not in_position:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B (hedged by hedge_ratio)
                return (shares_b, 0, 0)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                # Long Asset A
                return (shares_a, 0, 0)
            else:
                # Short Asset B
                return (-shares_b, 0, 0)

        # No entry
        return (np.nan, 0, 0)

    # If already in position, decide exits
    # STOP-LOSS: absolute z-score exceeded
    if abs(z) > stop_threshold:
        # Close this asset's position
        return (-pos, 0, 0)

    # EXIT on mean reversion: z crosses zero or is within exit_threshold
    # Use previous z to detect crossing if available
    prev_z = zscore[i - 1] if i > 0 else np.nan
    crossed_zero = False
    if not np.isnan(prev_z):
        if prev_z * z < 0:
            crossed_zero = True

    if crossed_zero or abs(z) <= exit_threshold:
        return (-pos, 0, 0)

    # Otherwise hold position
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: Union[pd.DataFrame, pd.Series, np.ndarray],
    asset_b: Union[pd.DataFrame, pd.Series, np.ndarray],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators for the pairs strategy.

    Accepts either DataFrames (with 'close' column), Series, or numpy arrays.

    Returns dictionary with keys:
      - 'close_a': np.ndarray
      - 'close_b': np.ndarray
      - 'hedge_ratio': np.ndarray
      - 'zscore': np.ndarray

    Implementation notes:
      - Rolling OLS hedge ratio is computed without lookahead: hedge_ratio[i]
        is estimated using data strictly before index i (window ends at i-1).
      - For early bars where full lookback is not available, regression uses the
        available past samples (minimum 2 points). This prevents long NaN
        regions and helps satisfy warmup requirements.
      - Rolling mean/std for z-score use min_periods=1 and ddof=0 to avoid NaNs
        for small samples. If std == 0, z-score is set to 0.
    """
    # Extract close arrays from possible input types
    def _extract_close(x: Union[pd.DataFrame, pd.Series, np.ndarray], name: str) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError(f"DataFrame for {name} must contain 'close' column")
            arr = x['close'].values
        elif isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            # Try to convert to numpy array
            arr = np.asarray(x)
        # Ensure 1-D float array
        arr = arr.astype('float64').flatten()
        return arr

    close_a = _extract_close(asset_a, 'asset_a')
    close_b = _extract_close(asset_b, 'asset_b')

    if len(close_a) != len(close_b):
        raise ValueError('asset_a and asset_b must have the same length')

    n = len(close_a)
    hedge_ratio = np.full(n, np.nan, dtype='float64')

    # Rolling OLS regression (A ~ slope * B) using only past data up to i-1
    # For i from 1..n-1, use window = min(hedge_lookback, i)
    for i in range(1, n):
        start = max(0, i - hedge_lookback)
        x = close_b[start:i]  # excludes current index i
        y = close_a[start:i]
        # Remove NaN pairs
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() >= 2:
            # Compute OLS slope; stats.linregress requires finite arrays
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                hedge_ratio[i] = float(slope)
            except Exception:
                # In case regression fails for numerical reasons, leave NaN
                hedge_ratio[i] = np.nan
        else:
            hedge_ratio[i] = np.nan

    # For index 0 hedge_ratio remains NaN. For stability, forward-fill with first valid slope
    # using previous values only (no future lookahead): we can carry last valid slope forward.
    last_valid = np.nan
    for i in range(n):
        if not np.isnan(hedge_ratio[i]):
            last_valid = hedge_ratio[i]
        else:
            # keep NaN until we have at least one valid slope
            if last_valid is not None and not np.isnan(last_valid):
                hedge_ratio[i] = last_valid

    # Compute spread using hedge_ratio aligned so that hedge_ratio[i] uses past data
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score: use min_periods=1 and ddof=0 to avoid NaNs
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=1).mean().to_numpy()
    # Use population std (ddof=0) so single-sample std = 0 (not NaN)
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=1).std(ddof=0).to_numpy()

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = np.where(spread_std > 0, (spread - spread_mean) / spread_std, 0.0)

    # Ensure outputs are numpy arrays of correct dtype
    return {
        'close_a': np.asarray(close_a, dtype='float64'),
        'close_b': np.asarray(close_b, dtype='float64'),
        'hedge_ratio': np.asarray(hedge_ratio, dtype='float64'),
        'zscore': np.asarray(zscore, dtype='float64'),
    }
