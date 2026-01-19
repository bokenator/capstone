import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Dict, Tuple, Any


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0
) -> tuple:
    """
    Generate orders for pairs trading. Designed for flexible multi-asset mode.

    Note: The wrapper used in the backtest calls this function with the
    arguments in the order (c, close_a, close_b, zscore, hedge_ratio, ...).

    Args:
        c: OrderContext-like object with attributes i (index), col (asset column),
           position_now (current position for this asset), cash_now (available cash)
        close_a: 1D array of Asset A close prices
        close_b: 1D array of Asset B close prices
        zscore: 1D array of spread z-score values
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: z-score level to enter (positive)
        exit_threshold: z-score level to exit (typically 0.0)
        stop_threshold: z-score level to stop-loss (positive)
        notional_per_leg: fixed notional per leg in dollars (default 10_000)

    Returns:
        (size, size_type, direction) tuple as required by the testing wrapper:
        - size: number of shares (positive=buy, negative=sell)
        - size_type: 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: 0=Both, 1=LongOnly, 2=ShortOnly

    Behavior:
        - Entry: z > entry_threshold -> Short A, Long B
                 z < -entry_threshold -> Long A, Short B
        - Exit: z crosses zero (sign change) OR abs(z) > stop_threshold -> close
        - Position sizing: target shares for Asset A = notional_per_leg / price_a
          Asset B shares = hedge_ratio * shares_a (so positions in shares follow hedge ratio)
    """
    i: int = int(c.i)
    col: int = int(c.col)
    pos_now: float = float(getattr(c, 'position_now', 0.0))

    # Safety checks
    if i < 0:
        return (0.0, 0, 0)

    # Bounds check for arrays
    n = len(close_a)
    if i >= n or i >= len(close_b) or i >= len(zscore) or i >= len(hedge_ratio):
        return (0.0, 0, 0)

    price_a = float(close_a[i])
    price_b = float(close_b[i])
    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If any vital data is NaN, do nothing
    if np.isnan(price_a) or np.isnan(price_b) or np.isnan(z) or np.isnan(hr):
        return (0.0, 0, 0)

    # Compute target shares based on notional per leg and hedge ratio in shares
    shares_a = 0.0
    shares_b = 0.0
    if price_a > 0:
        shares_a = float(notional_per_leg / price_a)
    if not np.isnan(hr):
        shares_b = float(hr * shares_a)

    eps = 1e-8
    in_position = abs(pos_now) > eps

    # Previous z-score for crossing detection
    z_prev = zscore[i - 1] if i > 0 else np.nan

    # Determine exit conditions (only relevant if we currently hold a position)
    if in_position:
        # Stop-loss: absolute z-score exceeds stop threshold
        if abs(z) > stop_threshold:
            # Close entire position for this asset
            return (-pos_now, 0, 0)

        # Exit on mean reversion (z-score crosses zero)
        if not np.isnan(z_prev):
            if (z_prev > 0 and z <= exit_threshold) or (z_prev < 0 and z >= exit_threshold):
                return (-pos_now, 0, 0)
        else:
            # Fallback: if z is within exit threshold magnitude
            if abs(z) <= abs(exit_threshold):
                return (-pos_now, 0, 0)

    # Entry conditions (only if not currently in position for this asset)
    if not in_position:
        # Short Asset A, Long Asset B when z > entry threshold
        if z > entry_threshold:
            if col == 0:
                # Short Asset A
                return (-shares_a, 0, 0)
            else:
                # Long Asset B
                return (shares_b, 0, 0)

        # Long Asset A, Short Asset B when z < -entry threshold
        if z < -entry_threshold:
            if col == 0:
                return (shares_a, 0, 0)
            else:
                return (-shares_b, 0, 0)

    # No action -> return zero-sized order (non-NaN)
    return (0.0, 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score for a pair of assets.

    The function accepts pandas DataFrames/Series or numpy arrays and always
    returns a dict of numpy arrays with keys: 'close_a', 'close_b', 'hedge_ratio', 'zscore'.

    All returned arrays are aligned and trimmed to the shortest available length.
    After a modest warmup, NaNs are forward-filled (using only past information)
    and any remaining early NaNs are set to 0.0 to avoid NaNs in downstream tests.
    """
    # Helper to extract close arrays from different input types
    def _extract_close(obj):
        if isinstance(obj, pd.DataFrame):
            if 'close' in obj.columns:
                return obj['close'].astype(float).values
            elif obj.shape[1] == 1:
                return obj.iloc[:, 0].astype(float).values
            else:
                raise ValueError("asset DataFrame must contain a 'close' column or be a single-column frame")
        elif isinstance(obj, pd.Series):
            return obj.astype(float).values
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return np.asarray(obj, dtype=float)
        else:
            raise TypeError("Unsupported asset type for close price extraction")

    # Support case where the first argument is a combined DataFrame with both assets
    if isinstance(asset_a, pd.DataFrame) and set(['asset_a', 'asset_b']).issubset(asset_a.columns):
        close_a = _extract_close(asset_a['asset_a'])
        close_b = _extract_close(asset_a['asset_b'])
    else:
        close_a = _extract_close(asset_a)
        close_b = _extract_close(asset_b)

    # Align lengths to the shortest of the two
    if len(close_a) != len(close_b):
        min_len = min(len(close_a), len(close_b))
        if min_len == 0:
            raise ValueError('One of the input price series is empty')
        close_a = close_a[:min_len]
        close_b = close_b[:min_len]

    n = len(close_a)

    # Compute rolling hedge ratio using OLS on y = close_a, x = close_b
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Minimum regression window: use at least 2, but prefer 20 if hedge_lookback is larger
    min_regress = int(min(max(20, 2), hedge_lookback))
    min_regress = max(2, min_regress)

    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        x = close_b[start:i + 1]
        y = close_a[start:i + 1]

        # Remove NaNs
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() >= min_regress:
            x_valid = x[mask]
            y_valid = y[mask]
            # Compute slope via linear regression
            try:
                slope, intercept, r, p, stderr = stats.linregress(x_valid, y_valid)
                if not np.isnan(slope):
                    hedge_ratio[i] = float(slope)
            except Exception:
                hedge_ratio[i] = np.nan

    # Forward-fill hedge ratio (uses only past information)
    hr_s = pd.Series(hedge_ratio)
    hr_ffill = hr_s.fillna(method='ffill')
    hr_final = hr_ffill.fillna(0.0).values

    # Compute spread and rolling z-score
    spread = pd.Series(close_a) - pd.Series(hr_final) * pd.Series(close_b)
    spread_mean = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).mean()
    spread_std = spread.rolling(window=zscore_lookback, min_periods=zscore_lookback).std()

    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    zscore_ffill = zscore.fillna(method='ffill')
    zscore_final = zscore_ffill.fillna(0.0).values

    # Ensure no NaNs remain after warmup (50) by forward-filling the tail only
    warmup = 50
    if n > warmup:
        # Use only past information to fill the tail for both series
        hr_tail = pd.Series(hr_final).iloc[warmup:].fillna(method='ffill').fillna(0.0).values
        zs_tail = pd.Series(zscore_final).iloc[warmup:].fillna(method='ffill').fillna(0.0).values

        hr_final[warmup:] = hr_tail
        zscore_final[warmup:] = zs_tail

    return {
        'close_a': np.asarray(close_a, dtype=float),
        'close_b': np.asarray(close_b, dtype=float),
        'hedge_ratio': np.asarray(hr_final, dtype=float),
        'zscore': np.asarray(zscore_final, dtype=float),
    }
