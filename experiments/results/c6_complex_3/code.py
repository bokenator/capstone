"""
Pairs trading strategy implementation for vectorbt backtest.
Exports:
- compute_spread_indicators(close_a, close_b, hedge_lookback=60, zscore_lookback=20) -> dict with 'zscore' and 'hedge_ratio' (np.ndarray)
- order_func(c, close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold) -> (size, size_type, direction)

Notes:
- Uses rolling OLS for hedge ratio (scipy.stats.linregress) with a variable window up to hedge_lookback (uses available data when fewer points exist).
- Rolling z-score computed with pandas rolling (min_periods=1, std ddof=0) to avoid NaNs; if std is zero or non-finite z-score is set to 0.
- Order function implements the described logic: entry when z-score crosses entry_threshold (reduces repeated order spam), exit on z crossing 0 or stop-loss |z|>stop_threshold.
- Position sizing: fixed base notional per leg = $10,000; asset B units scaled by hedge_ratio * base_units (so position_b ~= hedge_ratio * position_a).

This file only uses APIs from the Verified API Surface.
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy.stats

# Try to import enums from vectorbt if available; fall back to sensible integer defaults
try:
    from vectorbt.portfolio.enums import Direction as _VBT_Direction, SizeType as _VBT_SizeType
except Exception:
    _VBT_Direction = None
    _VBT_SizeType = None

# Track last order bar per column to avoid emitting duplicate orders for the same asset and bar
_LAST_ORDER_BAR: dict = {}


def _to_series_like(x: Any) -> pd.Series:
    """Convert array-like or pandas objects to a 1D pandas Series.

    Accepts numpy arrays, lists, pandas Series, pandas DataFrame (takes first column).
    """
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame):
        # take first column
        return x.iloc[:, 0].copy()
    # For numpy arrays / lists: build DataFrame then take first column to handle shapes like (n,1)
    return pd.DataFrame(x).iloc[:, 0]


def _resolve_order_enums() -> Tuple[int, int, Optional[int], Optional[int], str]:
    """Resolve integer values for BUY/SELL directions and size type codes.

    Returns (DIR_BUY, DIR_SELL, SIZE_TYPE_TARGET, SIZE_TYPE_AMOUNT, SIZE_TYPE_NAME)
    SIZE_TYPE_TARGET is preferred for setting a target position (if available).
    SIZE_TYPE_AMOUNT is preferred for specifying a trade amount (if available).
    SIZE_TYPE_NAME is the name chosen for the target-type (or empty string if none).
    """
    DIR_BUY = 1
    DIR_SELL = 2
    SIZE_TYPE_TARGET: Optional[int] = None
    SIZE_TYPE_AMOUNT: Optional[int] = None
    SIZE_TYPE_NAME = ""

    try:
        if _VBT_Direction is not None:
            if hasattr(_VBT_Direction, 'BUY') and hasattr(_VBT_Direction, 'SELL'):
                DIR_BUY = int(getattr(_VBT_Direction, 'BUY'))
                DIR_SELL = int(getattr(_VBT_Direction, 'SELL'))
            elif hasattr(_VBT_Direction, 'LONG') and hasattr(_VBT_Direction, 'SHORT'):
                DIR_BUY = int(getattr(_VBT_Direction, 'LONG'))
                DIR_SELL = int(getattr(_VBT_Direction, 'SHORT'))
    except Exception:
        DIR_BUY, DIR_SELL = 1, 2

    try:
        if _VBT_SizeType is not None:
            # Find an amount-style size type first (units)
            for name in ('Size', 'SIZE', 'Amount', 'AMOUNT'):
                if hasattr(_VBT_SizeType, name):
                    SIZE_TYPE_AMOUNT = int(getattr(_VBT_SizeType, name))
                    break

            # Prefer 'Target' for target orders if available
            if hasattr(_VBT_SizeType, 'Target'):
                SIZE_TYPE_TARGET = int(getattr(_VBT_SizeType, 'Target'))
                SIZE_TYPE_NAME = 'Target'
            elif hasattr(_VBT_SizeType, 'TARGET'):
                SIZE_TYPE_TARGET = int(getattr(_VBT_SizeType, 'TARGET'))
                SIZE_TYPE_NAME = 'TARGET'

            # If we didn't find a target-type, but found an amount-type, use amount-type as target fallback
            if SIZE_TYPE_TARGET is None and SIZE_TYPE_AMOUNT is not None:
                SIZE_TYPE_TARGET = SIZE_TYPE_AMOUNT
                SIZE_TYPE_NAME = 'AmountFallback'
    except Exception:
        SIZE_TYPE_TARGET = SIZE_TYPE_AMOUNT = None
        SIZE_TYPE_NAME = ""

    return DIR_BUY, DIR_SELL, SIZE_TYPE_TARGET, SIZE_TYPE_AMOUNT, SIZE_TYPE_NAME

# Resolve enum values once
_DIR_BUY, _DIR_SELL, _SIZE_TYPE_TARGET, _SIZE_TYPE_AMOUNT, _SIZE_TYPE_NAME = _resolve_order_enums()


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling hedge ratio and z-score for the spread between two assets.

    Args:
        close_a: Prices for asset A (1d array-like)
        close_b: Prices for asset B (1d array-like)
        hedge_lookback: Lookback window (in bars) for rolling OLS hedge ratio
        zscore_lookback: Lookback window for rolling mean/std used in z-score

    Returns:
        Dict with keys:
            - 'zscore': numpy array of z-scores (same length as inputs)
            - 'hedge_ratio': numpy array of hedge ratios (same length as inputs)
    """
    # Convert inputs to pandas Series safely (accept numpy arrays, pandas Series/DataFrame, lists, etc.)
    price_a_series = _to_series_like(close_a)
    price_b_series = _to_series_like(close_b)

    # Align series on common index to avoid lookahead when one input is longer than the other
    common_index = price_a_series.index.intersection(price_b_series.index)
    if len(common_index) == 0:
        # fallback: align by truncating to the minimum length
        m = min(len(price_a_series), len(price_b_series))
        price_a_series = price_a_series.iloc[:m]
        price_b_series = price_b_series.iloc[:m]
    else:
        price_a_series = price_a_series.loc[common_index]
        price_b_series = price_b_series.loc[common_index]

    n = len(price_a_series)

    # Hedge ratio: rolling OLS slope of (A ~ B)
    hedge_ratio = np.full(n, np.nan, dtype=np.float64)

    last_slope = 1.0
    for i in range(n):
        start = 0 if i - hedge_lookback + 1 < 0 else (i - hedge_lookback + 1)
        y_win = price_a_series.values[start : i + 1]
        x_win = price_b_series.values[start : i + 1]

        # Mask out non-finite values pairwise
        mask = np.isfinite(y_win) & np.isfinite(x_win)
        if np.sum(mask) >= 2:
            try:
                res = scipy.stats.linregress(x_win[mask], y_win[mask])
                slope = float(res.slope) if np.isfinite(res.slope) else last_slope
            except Exception:
                slope = last_slope
        else:
            # Not enough data for regression: use last known slope
            slope = last_slope

        hedge_ratio[i] = slope
        last_slope = slope

    # Compute spread using the hedge ratio (element-wise)
    spread = price_a_series.values - hedge_ratio * price_b_series.values

    # Rolling mean and std for z-score (use min_periods=1 and ddof=0 for std to avoid NaNs early on)
    spread_series = pd.Series(spread, index=price_a_series.index)
    rolling_mean_series = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).mean()
    # Use ddof=0 so std of single-point window is 0 (not NaN) where supported
    try:
        rolling_std_series = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std(ddof=0)
    except TypeError:
        # Older pandas versions may not accept ddof in rolling.std(); fall back to default and handle NaNs
        rolling_std_series = pd.Series.rolling(spread_series, window=zscore_lookback, min_periods=1).std()

    mean_np = rolling_mean_series.values
    std_np = rolling_std_series.values

    # Compute z-score safely: if std is zero or non-finite, set z-score to 0.0
    zscore = np.zeros(n, dtype=np.float64)
    finite_std_mask = np.isfinite(std_np) & (std_np > 1e-12)
    zscore[finite_std_mask] = (spread[finite_std_mask] - mean_np[finite_std_mask]) / std_np[finite_std_mask]

    # Ensure hedge_ratio is finite everywhere: replace non-finite with previous or 1.0
    hr = hedge_ratio.copy()
    for i in range(n):
        if not np.isfinite(hr[i]):
            hr[i] = hr[i - 1] if i > 0 and np.isfinite(hr[i - 1]) else 1.0

    return {
        "zscore": zscore,
        "hedge_ratio": hr,
    }


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, int, int]:
    """Order function for flexible multi-asset vectorbt backtest.

    Expected signature by the test harness wrapper. The wrapper simulates an OrderContext 'c'
    with attributes i (bar index), col (asset column 0 or 1), position_now (current position in units),
    and cash_now/value_now.

    Returns a tuple: (size, size_type, direction).
    If no order should be placed, returns (np.nan, 0, 0) (wrapper treats NaN size as no order).

    Logic implemented:
      - Entry when zscore crosses entry_threshold (reduces repeated order spam)
      - Exit when zscore crosses 0 or |z| > stop_threshold
      - Position sizing: base notional per leg = $10,000; asset A base_units = notional / price_a; asset B units = hedge_ratio * base_units
    """
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Current position for this asset (in units)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Avoid emitting duplicate orders for the same asset and bar
    last_bar_for_col = _LAST_ORDER_BAR.get(col, -1)
    if last_bar_for_col == i:
        return (np.nan, 0, 0)

    # Safely extract prices and indicators at current bar using DataFrame->Series conversion
    price_a = float(_to_series_like(close_a).iloc[i])
    price_b = float(_to_series_like(close_b).iloc[i])

    z_t = float(zscore[i]) if (i >= 0 and i < len(zscore)) else 0.0
    hr_t = float(hedge_ratio[i]) if (i >= 0 and i < len(hedge_ratio)) else 1.0

    # Previous z-score for zero-cross detection
    z_prev = float(zscore[i - 1]) if i > 0 else z_t

    # Base notional per leg
    NOTIONAL_PER_LEG = 10_000.0

    # Compute base units using asset A price as reference
    base_units = NOTIONAL_PER_LEG / price_a if price_a > 0 and np.isfinite(price_a) else 0.0

    # Units for this asset depending on column
    if col == 0:
        # Asset A
        target_units_entry = base_units
    else:
        # Asset B scaled by hedge ratio
        target_units_entry = max(0.0, hr_t) * base_units

    # Determine signals (use crossing for entries to avoid repeat orders)
    enter_short = (z_prev <= float(entry_threshold)) and (z_t > float(entry_threshold))
    enter_long = (z_prev >= -float(entry_threshold)) and (z_t < -float(entry_threshold))
    stop_loss = np.abs(z_t) > float(stop_threshold)
    cross_zero = (z_prev < 0 and z_t >= 0) or (z_prev > 0 and z_t <= 0)

    # Closing condition: either stop_loss or crossing zero
    should_close = stop_loss or cross_zero

    # No-op default
    no_order = (np.nan, 0, 0)

    # Helper: whether we have a target-type and amount-type size codes
    have_target = _SIZE_TYPE_TARGET is not None
    have_amount = _SIZE_TYPE_AMOUNT is not None

    # If currently in position, check for exit conditions
    if pos_now != 0.0:
        if should_close:
            # Prefer amount-style closing (sell/buy the absolute position) if available to avoid zero-size target orders
            if have_amount:
                size = float(np.abs(pos_now))
                direction = _DIR_SELL if pos_now > 0 else _DIR_BUY
                _LAST_ORDER_BAR[col] = i
                return (size, _SIZE_TYPE_AMOUNT, int(direction))
            elif have_target:
                # Fall back to amount semantics using the target code to avoid zero-size orders
                size = float(np.abs(pos_now))
                direction = _DIR_SELL if pos_now > 0 else _DIR_BUY
                _LAST_ORDER_BAR[col] = i
                return (size, _SIZE_TYPE_TARGET, int(direction))
            else:
                # Last resort: use amount with generic code
                size = float(np.abs(pos_now))
                _LAST_ORDER_BAR[col] = i
                return (size, 1, int(_DIR_SELL if pos_now > 0 else _DIR_BUY))
        else:
            # No change to existing position
            return no_order

    # If not in a position, check for entry signals
    if pos_now == 0.0:
        # Entry: z > entry_threshold => Short A, Long B (on crossing)
        if enter_short:
            if col == 0:
                # Short A: prefer amount-type entry if available (place units)
                if have_amount:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_AMOUNT, int(_DIR_SELL))
                elif have_target:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_TARGET, int(_DIR_SELL))
                else:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), 1, int(_DIR_SELL))
            else:
                # Long B
                if have_amount:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_AMOUNT, int(_DIR_BUY))
                elif have_target:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_TARGET, int(_DIR_BUY))
                else:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), 1, int(_DIR_BUY))

        # Entry: z < -entry_threshold => Long A, Short B (on crossing)
        if enter_long:
            if col == 0:
                # Long A
                if have_amount:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_AMOUNT, int(_DIR_BUY))
                elif have_target:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_TARGET, int(_DIR_BUY))
                else:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), 1, int(_DIR_BUY))
            else:
                # Short B
                if have_amount:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_AMOUNT, int(_DIR_SELL))
                elif have_target:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), _SIZE_TYPE_TARGET, int(_DIR_SELL))
                else:
                    _LAST_ORDER_BAR[col] = i
                    return (float(target_units_entry), 1, int(_DIR_SELL))

    # Otherwise, no order
    return no_order
