import numpy as np
import pandas as pd
import scipy
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction
from typing import Any, Dict, Tuple


def _to_1d_series(x) -> pd.Series:
    """Utility: convert input to 1D pandas Series safely.

    Handles:
      - pd.Series -> returned copy
      - pd.DataFrame -> if 'close' column exists use it, else if single column use it, else take first column
      - numpy arrays (1D or 2D) -> flattened/squeezed to 1D
      - other array-like -> converted to numpy and squeezed
    """
    if isinstance(x, pd.Series):
        return x.copy()

    if isinstance(x, pd.DataFrame):
        # Prefer a column named 'close' if present
        if "close" in x.columns:
            return x["close"].copy()
        # If single-column DF, take that column
        if x.shape[1] == 1:
            return x.iloc[:, 0].copy()
        # Otherwise, take the first column
        return x.iloc[:, 0].copy()

    arr = np.array(x)
    # If 2D with one column, squeeze to 1D
    if arr.ndim > 1:
        try:
            arr = arr.squeeze()
        except Exception:
            arr = arr.reshape(-1)
    return pd.Series(arr)


def _get_direction_values_extended() -> Tuple[int, int, int]:
    """Extract numeric values for long, short and a neutral/both direction from Direction enum.

    Returns (dir_long, dir_short, dir_neutral). Uses heuristics and falls back to (1,2,0).
    """
    dir_long = None
    dir_short = None
    dir_neutral = None
    try:
        for member in Direction:
            name = getattr(member, 'name', str(member)).lower()
            if 'long' in name or name == 'buy' or name == 'b':
                dir_long = int(member)
            elif 'short' in name or name == 'sell' or name == 's':
                dir_short = int(member)
            elif 'both' in name or 'close' in name or 'either' in name or 'any' in name or 'neutral' in name:
                dir_neutral = int(member)
        # Heuristics if values still not found
        members = list(Direction)
        if dir_long is None and len(members) >= 1:
            dir_long = int(members[0])
        if dir_short is None and len(members) >= 2:
            dir_short = int(members[1])
        if dir_neutral is None and len(members) >= 3:
            dir_neutral = int(members[2])
    except Exception:
        pass

    if dir_long is None:
        dir_long = 1
    if dir_short is None:
        dir_short = 2
    if dir_neutral is None:
        dir_neutral = 0

    return int(dir_long), int(dir_short), int(dir_neutral)


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and z-score of the spread between two assets.

    This function is defensive about input types and lengths:
      - Accepts numpy arrays, pandas Series, or DataFrames.
      - Aligns asset B to the index of asset A so the output length matches the first input.

    Args:
        close_a (array-like): Close prices of asset A.
        close_b (array-like): Close prices of asset B.
        hedge_lookback (int): Lookback window for rolling OLS regression (max window).
        zscore_lookback (int): Lookback window for rolling mean/std of spread.

    Returns:
        dict: {
            "zscore": np.ndarray of z-scores (same length as first input),
            "hedge_ratio": np.ndarray of hedge ratios (same length as first input)
        }
    """
    # Convert to 1D pandas Series to handle alignment and indexing robustly
    a_series = _to_1d_series(close_a)
    b_series = _to_1d_series(close_b)

    # Reindex b to match the index of a. This ensures output length equals len(close_a).
    try:
        b_series = b_series.reindex(a_series.index)
    except Exception:
        # If reindexing fails for some reason, fall back to trimming/padding by position
        min_len = min(len(a_series), len(b_series))
        a_series = a_series.iloc[:min_len]
        b_series = b_series.iloc[:min_len]

    a = a_series.values.astype(float)
    b = b_series.values.astype(float)

    n = a.shape[0]

    hedge_ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    zscore = np.full(n, np.nan, dtype=float)

    # Initial hedge ratio fallback (used until regression provides a value)
    last_hr = 1.0

    for i in range(n):
        # Rolling (or expanding if not enough data) regression for hedge ratio
        start = max(0, i - hedge_lookback + 1)
        x_win = b[start : i + 1]
        y_win = a[start : i + 1]

        # Use only finite values
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if np.sum(mask) >= 2:
            try:
                slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(x_win[mask], y_win[mask])
                if np.isfinite(slope):
                    hedge_ratio[i] = float(slope)
                    last_hr = float(slope)
                else:
                    hedge_ratio[i] = last_hr
            except Exception:
                # In case regression fails, carry forward last hedge ratio
                hedge_ratio[i] = last_hr
        else:
            # Not enough data points -> carry forward previous hedge ratio
            hedge_ratio[i] = last_hr

        # Compute spread for this bar
        if np.isfinite(hedge_ratio[i]) and np.isfinite(a[i]) and np.isfinite(b[i]):
            spread[i] = a[i] - hedge_ratio[i] * b[i]
        else:
            spread[i] = np.nan

        # Rolling mean/std for z-score using available window up to current index
        zs_start = max(0, i - zscore_lookback + 1)
        win_vals = spread[zs_start : i + 1]
        finite_vals = win_vals[np.isfinite(win_vals)]

        if finite_vals.size >= 1:
            mu = np.mean(finite_vals)
            sigma = np.std(finite_vals)
            # Protect against zero std
            if sigma == 0 or not np.isfinite(sigma):
                # If sigma is zero, z-score is defined as 0 (no dispersion)
                zscore[i] = 0.0
            else:
                # If current spread is nan, zscore becomes nan; protect that
                if np.isfinite(spread[i]):
                    zscore[i] = (spread[i] - mu) / sigma
                else:
                    zscore[i] = np.nan
        else:
            zscore[i] = np.nan

    return {"zscore": np.array(zscore, dtype=float), "hedge_ratio": np.array(hedge_ratio, dtype=float)}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading in flexible multi-asset mode.

    Returns a tuple (size, size_type, direction) or (np.nan, 0, 0) to indicate no order.

    Logic:
      - Entry when z-score crosses the entry threshold (reduces repeated orders while zscore remains beyond threshold).
      - Exit when zscore crosses 0 or |zscore| > stop_threshold.
      - Position sizing: fixed notional per leg (10,000 USD) for asset A; asset B sized by hedge ratio.
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))
    pos_now = float(getattr(c, "position_now", 0.0))

    # Safety checks
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    cur_z = zscore[i]
    hr = hedge_ratio[i] if i < len(hedge_ratio) else np.nan

    price = float(close_a[i]) if col == 0 else float(close_b[i])

    # If insufficient data or NaNs, do nothing
    if not np.isfinite(cur_z) or not np.isfinite(hr) or not np.isfinite(price) or price <= 0:
        return (np.nan, 0, 0)

    # Determine previous z-score to detect crossing
    prev_z = zscore[i - 1] if i > 0 else np.nan

    # Entry crossing detection (only enter when crossing threshold to avoid repeated orders)
    crossed_entry_pos = (cur_z > entry_threshold) and (not np.isfinite(prev_z) or prev_z <= entry_threshold)
    crossed_entry_neg = (cur_z < -entry_threshold) and (not np.isfinite(prev_z) or prev_z >= -entry_threshold)

    # Determine zero crossing for exits
    crossed_zero = False
    if i > 0 and np.isfinite(prev_z):
        if (prev_z < 0 and cur_z >= 0) or (prev_z > 0 and cur_z <= 0):
            crossed_zero = True

    # Interpret whether currently in position (small tolerance)
    in_position = abs(pos_now) > 1e-12

    # Fixed notional per leg (USD)
    NOTIONAL = 10_000.0

    # Compute desired unit sizes based on current prices and hedge ratio
    price_a = float(close_a[i])
    price_b = float(close_b[i])
    units_a = NOTIONAL / price_a if price_a > 0 and np.isfinite(price_a) else np.nan
    units_b = float(hr) * units_a if np.isfinite(hr) and np.isfinite(units_a) else np.nan

    # For value-based orders, compute value for each leg (USD)
    value_a = NOTIONAL
    value_b = abs(units_b) * price_b if np.isfinite(units_b) and np.isfinite(price_b) else np.nan

    # Size type (value) using the enum directly (preferred)
    try:
        SIZE_TYPE_VALUE = int(SizeType.Value)
    except Exception:
        # fallback to 1
        SIZE_TYPE_VALUE = 1

    DIR_LONG, DIR_SHORT, DIR_NEUTRAL = _get_direction_values_extended()

    # If currently in position, check exit conditions
    if in_position:
        # Stop-loss
        if abs(cur_z) > stop_threshold:
            # Close this asset by submitting an order equal to current position magnitude (value-based)
            size_value = abs(pos_now) * price if np.isfinite(price) else np.nan
            if not np.isfinite(size_value) or size_value <= 0:
                return (np.nan, 0, 0)
            # Prefer neutral direction for closing; fall back to opposite direction
            direction = DIR_NEUTRAL if DIR_NEUTRAL != 0 else (DIR_SHORT if pos_now > 0 else DIR_LONG)
            return (float(size_value), int(SIZE_TYPE_VALUE), int(direction))

        # Exit on mean reversion (zero crossing)
        if crossed_zero:
            size_value = abs(pos_now) * price if np.isfinite(price) else np.nan
            if not np.isfinite(size_value) or size_value <= 0:
                return (np.nan, 0, 0)
            direction = DIR_NEUTRAL if DIR_NEUTRAL != 0 else (DIR_SHORT if pos_now > 0 else DIR_LONG)
            return (float(size_value), int(SIZE_TYPE_VALUE), int(direction))

        # Otherwise hold
        return (np.nan, 0, 0)

    # Not in position -> check entry signals (require crossing)
    if crossed_entry_pos:
        # Short A, Long B
        if col == 0:
            # Asset A -> short fixed notional (value order)
            if not np.isfinite(value_a) or value_a <= 0:
                return (np.nan, 0, 0)
            return (float(value_a), int(SIZE_TYPE_VALUE), int(DIR_SHORT))
        else:
            # Asset B -> sign determined by units_b; use value-based order
            if not np.isfinite(value_b) or value_b <= 0:
                return (np.nan, 0, 0)
            dir_b = DIR_LONG if units_b > 0 else DIR_SHORT
            return (float(value_b), int(SIZE_TYPE_VALUE), int(dir_b))

    if crossed_entry_neg:
        # Long A, Short B
        if col == 0:
            if not np.isfinite(value_a) or value_a <= 0:
                return (np.nan, 0, 0)
            return (float(value_a), int(SIZE_TYPE_VALUE), int(DIR_LONG))
        else:
            if not np.isfinite(value_b) or value_b <= 0:
                return (np.nan, 0, 0)
            dir_b = DIR_SHORT if units_b > 0 else DIR_LONG
            return (float(value_b), int(SIZE_TYPE_VALUE), int(dir_b))

    # No action
    return (np.nan, 0, 0)
