import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats


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

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    """
    # Extract context
    i = int(c.i)
    col = int(c.col)
    pos = float(getattr(c, 'position_now', 0.0))

    # Basic bounds checks
    if i < 0:
        return (np.nan, 0, 0)

    # Guard array bounds
    n = len(zscore)
    if i >= n:
        return (np.nan, 0, 0)

    z = float(zscore[i]) if not np.isnan(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if i < len(hedge_ratio) and not np.isnan(hedge_ratio[i]) else np.nan

    # If we don't have a valid z-score or hedge ratio, do nothing
    if np.isnan(z) or np.isnan(hr):
        return (np.nan, 0, 0)

    # Prices
    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan
    if not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Determine base share sizing (use Asset A notional as base)
    shares_a = notional_per_leg / price_a
    # Apply hedge ratio to get units of B per unit of A
    shares_b = shares_a * hr

    # Helper to compute target and create order (amount-based)
    def make_order(target: float) -> tuple:
        # target is desired position in shares for this asset
        size = target - pos
        # If size is effectively zero, don't place order
        if abs(size) < 1e-8:
            return (np.nan, 0, 0)
        return (float(size), 0, 0)

    # Stop-loss: if z-score exceeds stop threshold, close any open position
    if abs(z) > stop_threshold and pos != 0:
        return make_order(0.0)

    # Exit: close when z-score crosses the exit threshold (commonly zero)
    if pos != 0:
        crossed = False
        if i > 0 and not np.isnan(zscore[i - 1]):
            prev_z = float(zscore[i - 1])
            # For exit_threshold == 0, detect sign crossing
            if exit_threshold == 0.0:
                if prev_z * z <= 0:
                    crossed = True
            else:
                if (prev_z > exit_threshold and z <= exit_threshold) or (prev_z < -exit_threshold and z >= -exit_threshold):
                    crossed = True
        if crossed:
            return make_order(0.0)

    # Entry: only enter if currently flat for this asset
    if pos == 0:
        # Short A, Long B when z > entry_threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return make_order(-shares_a)
            else:
                # Long Asset B scaled by hedge ratio
                return make_order(shares_b)

        # Long A, Short B when z < -entry_threshold
        if z < -entry_threshold:
            if col == 0:
                return make_order(shares_a)
            else:
                return make_order(-shares_b)

    # No action by default
    return (np.nan, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Precompute all indicators for pairs strategy.

    Accepts either pandas DataFrame/Series with a 'close' column or raw numpy arrays.

    Args:
        asset_a: DataFrame/Series/ndarray with 'close' prices for Asset A
        asset_b: DataFrame/Series/ndarray with 'close' prices for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Convert inputs to numpy arrays (support series/dataframe/ndarray)
    def _extract_close(x):
        if isinstance(x, pd.DataFrame):
            if 'close' not in x.columns:
                raise KeyError("DataFrame input must contain 'close' column")
            arr = x['close'].values.astype(float)
        elif isinstance(x, pd.Series):
            arr = x.values.astype(float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        else:
            # Try to coerce
            arr = np.asarray(x, dtype=float)
        return arr

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    if len(close_a) != len(close_b):
        raise ValueError('Asset arrays must have the same length')

    n = len(close_a)

    # Validate lookbacks
    hedge_lookback = int(hedge_lookback)
    zscore_lookback = int(zscore_lookback)
    if hedge_lookback < 2:
        raise ValueError('hedge_lookback must be >= 2')
    if zscore_lookback < 1:
        raise ValueError('zscore_lookback must be >= 1')

    # Rolling hedge ratio (OLS slope of regressing A ~ B using past-inclusive window)
    # Use a window that ends at current index i (inclusive) to avoid lookahead while producing early values
    hedge_ratio = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        end = i + 1
        y = close_a[start:end]
        x = close_b[start:end]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            hedge_ratio[i] = np.nan
            continue
        try:
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread (uses hedge ratio available at time i)
    spread = np.full(n, np.nan, dtype=float)
    for i in range(n):
        hr = hedge_ratio[i]
        if not np.isfinite(hr):
            spread[i] = np.nan
        else:
            if np.isfinite(close_a[i]) and np.isfinite(close_b[i]):
                spread[i] = close_a[i] - hr * close_b[i]
            else:
                spread[i] = np.nan

    # Rolling mean and std of spread (uses past window including current index)
    spread_s = pd.Series(spread)
    spread_mean = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean().values
    # Use population std (ddof=0) to be stable for small windows
    spread_std = spread_s.rolling(window=zscore_lookback, min_periods=zscore_lookback).std(ddof=0).values

    # Compute z-score safely
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (spread - spread_mean) / spread_std
        zscore[~np.isfinite(zscore)] = np.nan

    return {
        'close_a': np.array(close_a, dtype=float),
        'close_b': np.array(close_b, dtype=float),
        'hedge_ratio': np.array(hedge_ratio, dtype=float),
        'zscore': np.array(zscore, dtype=float),
    }