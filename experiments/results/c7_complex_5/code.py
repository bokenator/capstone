import numpy as np
import pandas as pd
import vectorbt as vbt
import scipy


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float = 10000.0,
) -> tuple:
    """
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    NOTE: The flexible wrapper in the backtest passes arguments in the order
    (c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold).
    We therefore expect zscore before hedge_ratio here. The notional_per_leg has a default
    so the function can be called even if the wrapper does not provide it.

    Args:
        c: vectorbt OrderContext-like object with attributes i, col, position_now, cash_now
        close_a: 1D array of close prices for Asset A
        close_b: 1D array of close prices for Asset B
        zscore: 1D array of spread z-scores
        hedge_ratio: 1D array of rolling hedge ratios
        entry_threshold: entry z-score threshold (positive)
        exit_threshold: exit z-score threshold (usually 0.0)
        stop_threshold: stop-loss z-score threshold (positive)
        notional_per_leg: fixed notional per leg in dollars

    Returns:
        (size, size_type, direction) as a tuple where size is number of shares
        (positive = buy, negative = sell), size_type=0 (Amount), direction=0 (Both)

    Notes:
        - Uses simple share sizing: shares_a = notional_per_leg / price_a
          shares_b = hedge_ratio * shares_a so that position_b ≈ -hedge_ratio * position_a
        - All signals are computed without lookahead (reads zscore[i], hedge_ratio[i])
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Basic validation
    n = len(close_a)
    if i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Current raw values
    price_a = float(close_a[i])
    price_b = float(close_b[i])

    z = float(zscore[i]) if (i < len(zscore) and not np.isnan(zscore[i])) else np.nan
    hr = float(hedge_ratio[i]) if (i < len(hedge_ratio) and not np.isnan(hedge_ratio[i])) else np.nan

    # If indicators are not available or prices invalid, do nothing
    if np.isnan(z) or np.isnan(hr) or (not np.isfinite(price_a)) or (not np.isfinite(price_b)):
        return (np.nan, 0, 0)

    # Prevent division by zero or tiny prices
    if price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Compute share sizing: base shares for Asset A, scale Asset B by hedge ratio
    shares_a = float(notional_per_leg) / price_a
    shares_b = hr * shares_a

    # Previous z for exit crossing detection
    z_prev = float(zscore[i - 1]) if (i > 0 and not np.isnan(zscore[i - 1])) else np.nan

    # Determine targets for both assets
    target_a = None
    target_b = None

    # Stop-loss: absolute z beyond stop_threshold -> close both
    if np.abs(z) > stop_threshold:
        target_a = 0.0
        target_b = 0.0
    else:
        # Exit when z crosses exit_threshold (e.g., 0.0)
        crossed_to_exit = False
        if not np.isnan(z_prev):
            # crossing upward or downward through exit_threshold
            if (z_prev > exit_threshold and z <= exit_threshold) or (
                z_prev < exit_threshold and z >= exit_threshold
            ):
                crossed_to_exit = True

        if crossed_to_exit:
            target_a = 0.0
            target_b = 0.0
        else:
            # Entry logic
            if z > entry_threshold:
                # Short A, long B
                target_a = -shares_a
                target_b = shares_b
            elif z < -entry_threshold:
                # Long A, short B
                target_a = shares_a
                target_b = -shares_b
            else:
                # No signal to enter/exit
                return (np.nan, 0, 0)

    # Choose which asset we're creating an order for
    if col == 0:
        desired = float(target_a)
    else:
        desired = float(target_b)

    # Compute order size as delta from current position
    size = desired - pos_now

    # If size is effectively zero or not finite, do nothing
    if (not np.isfinite(size)) or (np.abs(size) <= 1e-8):
        return (np.nan, 0, 0)

    # Return size in shares (Amount), allow both directions
    return (float(size), 0, 0)


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> dict:
    """
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame or array-like with 'close' prices for Asset A (or a 1D array)
        asset_b: DataFrame or array-like with 'close' prices for Asset B (or a 1D array)
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays
    """
    # Extract close price arrays from inputs (support np.ndarray or pd.DataFrame/Series)
    # Helper to extract close series from various types
    def _extract_close(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if isinstance(x, pd.DataFrame):
                # If DataFrame has a 'close' column, use it
                if "close" in x.columns:
                    return x["close"].values
                # If DataFrame has exactly one column, use that
                if x.shape[1] == 1:
                    return x.iloc[:, 0].values
                # If DataFrame has exactly two columns and they are named asset_a/asset_b or similar,
                # do not decide here — caller should pass proper arguments. We'll fall back to values.
                return x.values
            else:
                return x.values
        else:
            # Array-like
            return np.array(x)

    close_a = _extract_close(asset_a)
    close_b = _extract_close(asset_b)

    # If either extraction yields a 2D array (e.g., whole DataFrame passed), try to handle common cases:
    # If close_a is 2D and close_b is 1D, assume close_a is a combined DataFrame with two columns and
    # take the first column as Asset A and second as Asset B (if close_b is an unrelated object, we'll align lengths below).
    if hasattr(close_a, "ndim") and getattr(close_a, "ndim") == 2:
        # If close_b is also 2D and shapes match, try to take corresponding columns
        if hasattr(close_b, "ndim") and getattr(close_b, "ndim") == 2 and close_a.shape == close_b.shape:
            # Both 2D with same shape: flatten columns by taking column 0 for A and column 1 for B
            close_a = np.array(close_a[:, 0]).flatten()
            close_b = np.array(close_b[:, 1]).flatten()
        else:
            # If close_a has exactly 2 columns, assume they are A and B
            if close_a.shape[1] >= 2:
                close_b_candidate = np.array(close_a[:, 1]).flatten()
                close_a = np.array(close_a[:, 0]).flatten()
                # If user also passed close_b, prefer the explicitly passed one but keep candidate otherwise
                if close_b is None or (hasattr(close_b, "shape") and getattr(close_b, "shape")[0] == 0):
                    close_b = close_b_candidate

    # Ensure numpy arrays
    close_a = np.array(close_a, dtype=float)
    close_b = np.array(close_b, dtype=float)

    # Align lengths to avoid lookahead when one input is truncated and the other is full-length
    if len(close_a) != len(close_b):
        min_len = int(min(len(close_a), len(close_b)))
        close_a = close_a[:min_len]
        close_b = close_b[:min_len]

    # Ensure same length after alignment
    if len(close_a) != len(close_b):
        raise ValueError("Input arrays must have the same length after alignment")

    n = len(close_a)

    # Prepare arrays
    hedge_ratio = np.full(n, np.nan)

    # Rolling OLS (no lookahead): compute slope using a rolling window that
    # ends at the current index i (inclusive). Use at least 2 points.
    for i in range(n):
        start = max(0, i - hedge_lookback + 1)
        end = i + 1
        if (end - start) < 2:
            continue

        x_win = close_b[start:end]
        y_win = close_a[start:end]

        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if np.sum(mask) < 2:
            continue

        try:
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x_win[mask], y_win[mask])
            hedge_ratio[i] = float(slope)
        except Exception:
            hedge_ratio[i] = np.nan

    # Spread using the estimated hedge_ratio (may be NaN for early bars)
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for z-score (uses past values up to current index)
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean().values
    rolling_std = pd.Series.rolling(spread_series, window=zscore_lookback).std().values

    # Compute z-score safely
    zscore = np.full(n, np.nan)
    valid = np.isfinite(rolling_std) & (rolling_std > 0)
    zscore[valid] = (spread[valid] - rolling_mean[valid]) / rolling_std[valid]

    return {
        "close_a": np.array(close_a, dtype=float),
        "close_b": np.array(close_b, dtype=float),
        "hedge_ratio": np.array(hedge_ratio, dtype=float),
        "zscore": np.array(zscore, dtype=float),
    }
